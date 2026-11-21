"""The forward post quotes the three result files its fortnight's read wrote."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
PAGE = ROOT / "posts" / "forward_2026_11" / "README.md"
MINUS = chr(0x2212)


def _signed(value: float, digits: int) -> str:
    text = f"{value:+.{digits}f}"
    return text.replace("-", MINUS)


def test_the_post_quotes_the_forward_read() -> None:
    page = " ".join(PAGE.read_text(encoding="utf-8").split(" ")).replace("\n", " ")
    lines = PAGE.read_text(encoding="utf-8")
    bench = json.loads((RESULTS / "bench_forward_2026_11.json").read_text(encoding="utf-8"))
    gate = json.loads((RESULTS / "regime_gate_G2.json").read_text(encoding="utf-8"))
    meta = json.loads((RESULTS / "reversion_meta_forward_2026_11.json").read_text(encoding="utf-8"))

    assert bench["f1_passes"] == [] and "all twelve killed" in page
    assert bench["p1"]["status"] == "not tested" and "**not tested**" in page
    assert gate["verdict"]["status"] == "killed" and gate["panel"]["paused_share"] == 0.0
    assert all(v["status"] == "killed" for v in meta["m1"].values()) and "all nine killed" in page

    rows = {r["name"]: r["bybit-2026-11"] for r in bench["rows"]}
    nets = [r["net_bp"] for r in rows.values()]
    assert f"{_signed(max(nets), 1)} to {_signed(min(nets), 1)} bp" in page.replace("from ", "")
    assert f"{bench['p1']['state']:.4f}".replace("-", MINUS) in page
    assert f"{rows['reversion']['trades']:,} trades at {_signed(rows['reversion']['net_bp'], 2)}" in page
    assert f"{_signed(gate['panel']['ungated_pct'], 2)}%" in page
    assert f"{rows['order_flow']['trades']:,}" in page and f"{rows['xgboost']['trades']:,}" in page

    plain = rows["reversion"]
    table = [("plain rule", plain)] + [
        (label, meta["scores"][name])
        for label, name in [
            ("logistic regression", "reversion_logistic"),
            ("XGBoost + kill switch", "reversion_meta_guarded"),
            ("GRU", "reversion_gru"),
            ("LightGBM", "reversion_lightgbm"),
            ("LSTM", "reversion_lstm"),
            ("TCN", "reversion_tcn"),
            ("XGBoost", "reversion_meta"),
            ("transformer", "reversion_transformer"),
            ("kill switch alone", "reversion_guarded"),
        ]
    ]
    for label, s in table:
        net = _signed(s["net_bp"], 1)
        net = f"**{net}**" if label == "logistic regression" else net
        line = f"| {label} | {s['trades']:,} | {_signed(s['gross_bp'], 1)} | {net} |"
        assert line in lines, line
    assert f"{_signed(meta['scores']['reversion_logistic']['net_bp'], 2)} bp" in page
