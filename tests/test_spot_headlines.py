"""The spot study's registered headline sentences, filled for every outcome."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest
import yaml

from trading_research.market_making.verdicts import KillCheck
from trading_research.spot import statuses as st

ROOT = Path(__file__).resolve().parents[1]
CONFIG: dict[str, Any] = yaml.safe_load((ROOT / "configs" / "spot_2026.yaml").read_text("utf-8"))
FALLBACK = "(H inside S25's information set)"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("tests do not reach the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)


def kill(
    name: str, value: float, reference: float, fired: bool = True, slot: str = "S25-BB"
) -> KillCheck:
    return KillCheck(slot, name, "test", value, reference, fired)


def test_the_templates_are_the_registered_sentences() -> None:
    assert CONFIG["artefacts"]["headline_sentences"] == st.TEMPLATES
    for name in st.TEMPLATES:
        spans = st._groups(st.TEMPLATES[name])
        filled = st.fill(name, *["v"] * len(spans))
        assert st.pattern(name).fullmatch(filled)


def test_each_slot_on_h() -> None:
    killed = st.slot_on_h("S25-BB", "killed", [kill("K1", -1.234, 0.0), kill("K-fee", 4.0, 10.0)])
    assert killed == (
        "S25-BB on H, Bybit spot BTCUSDT: killed; first kill K1 at -1.23 bp against 0.00 bp."
    )
    fee = st.slot_on_h("S25-PF", "killed", [kill("K1", 1.0, 0.0, False), kill("K-fee", 4.0, 10.0)])
    assert fee.endswith("first kill K-fee at 4.00 bp per side against 10.00 bp per side.")
    assert "Binance spot BTC/FDUSD" in fee
    label = st.slot_on_h("S25-T", "killed", [kill("K-label", -0.0123, 0.0)])
    assert label.endswith("K-label at -0.01 (rank IC) against 0.00 (rank IC).")
    days = st.slot_on_h("S25-T", "killed", [kill("K-days", 0.4, 0.5)])
    assert days.endswith(
        "K-days at 0.40 (share of days positive) against 0.50 (share of days positive)."
    )
    for status in ("candidate", f"candidate {FALLBACK}", "inconclusive (too few trades)", "void",
                   "not assessed (H unreadable on Bybit)", "not assessed (H read failed)"):  # fmt: skip
        sentence = st.slot_on_h("S25-BB", status, [kill("K1", 1.0, 0.0, False)])
        assert sentence.endswith(f": {status}; first kill none at n/a against n/a.")
    with pytest.raises(st.HeadlineError, match="no kill fired"):
        st.slot_on_h("S25-BB", "killed", [])
    with pytest.raises(st.HeadlineError, match="fired"):
        st.slot_on_h("S25-BB", "candidate", [kill("K1", -1.0, 0.0)])
    with pytest.raises(st.HeadlineError, match="registered status"):
        st.slot_on_h("S25-BB", "promising", [])
    with pytest.raises(st.HeadlineError, match="family"):
        st.slot_on_h("S25-PB", "killed", [kill("K1", -1.0, 0.0)])


def test_s25_pf_and_s25_t_sentences() -> None:
    pf = st.s25_pf_sentence(3.456, "killed")
    assert pf == (
        "S25's coordinates on nine print inputs, on Binance BTC/FDUSD, August to October 2026: "
        "gross edge 3.46 bp of turnover against the 10 bp taker fee a Regular account paid; "
        "break-even 3.46 bp per side; killed."
    )
    assert "gross edge n/a" in st.s25_pf_sentence(
        float("nan"), "not assessed (H unreadable on Binance)"
    )
    t = st.s25_t_sentence(87.25, "inconclusive (too few trades)")
    assert t == (
        "S25 as frozen on Bybit's perpetual, unchanged on Bybit's spot book: 87.2 trades a day "
        "against 240 registered; inconclusive (too few trades)."
    )


@pytest.mark.parametrize(
    ("buy", "sell", "end"),
    [
        (0.5, 0.25, "buys 0.50 bp and sells 0.25 bp better than the book walk."),
        (-0.5, -0.25, "buys 0.50 bp and sells 0.25 bp worse than the book walk."),
        (0.5, -0.25, "buys 0.50 bp and sells -0.25 bp better than the book walk."),
        (0.1, -0.75, "buys -0.10 bp and sells 0.75 bp worse than the book walk."),
        (0.0, 0.0, "buys 0.00 bp and sells 0.00 bp better than the book walk."),
        (-0.001, 0.0004, "buys 0.00 bp and sells 0.00 bp worse than the book walk."),
    ],
)
def test_the_calibration_sentence(buy: float, sell: float, end: str) -> None:
    sentence = st.calibration_sentence(buy, sell)
    assert sentence.startswith(
        "On Bybit's spot book, the print rule priced "
    ) and sentence.endswith(end)


def test_the_p_sentence_and_every_outcome() -> None:
    assert st.p_sentence("S25-PF", "confirmed") == "S25-PF on P: confirmed."
    h = {
        "S25-PF": st.Assessment("S25-PF", "H", "candidate"),
        "S25-BB": st.Assessment("S25-BB", "H", "killed", kills=[kill("K1", -2.0, 0.0)]),
        "S25-T": st.Assessment("S25-T", "H", "not assessed (H unreadable on Bybit)"),
    }
    out = st.headline_sentences(h, s25_pf_gross_bp_turnover=11.0, s25_t_trades_per_day=float("nan"),
                                calibration=None)  # fmt: skip
    assert len(out["each_slot_on_H"]) == 3 and out["calibration"] == [] and out["P"] == []
    assert out["S25-T"] == [st.s25_t_sentence(float("nan"), "not assessed (H unreadable on Bybit)")]
    later = st.headline_sentences(h, s25_pf_gross_bp_turnover=11.0, s25_t_trades_per_day=200.0,
                                  calibration=(0.1, 0.2), after_p={"S25-PF": "candidate (P not yet read)"})  # fmt: skip
    assert (
        later["P"] == ["S25-PF on P: candidate (P not yet read)."]
        and len(later["calibration"]) == 1
    )
    with pytest.raises(st.HeadlineError, match="no status after P"):
        st.headline_sentences(h, s25_pf_gross_bp_turnover=1.0, s25_t_trades_per_day=1.0, after_p={})
    with pytest.raises(st.HeadlineError, match="both sides"):
        st.calibration_sentence(float("nan"), 1.0)
