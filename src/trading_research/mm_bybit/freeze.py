"""The D amendment: the frozen values written into the configuration, and the entry.

:func:`frozen_values` gathers every placeholder's value from the stage records;
:func:`amend` writes them into the configuration (``config.freeze_text``: the
placeholders and nothing else), appends the dated entry with the frozen
configuration's sha256 to the document, and returns that hash.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from trading_research.market_making.prereg import sha256_bytes
from trading_research.mm_bybit import config


def frozen_values(
    stage0: dict[str, Any], w: dict[str, Any], d: dict[str, Any], compute: dict[str, Any]
) -> dict[str, Any]:
    admission = d.get("admission", {})
    return {
        "reader_test_result": stage0["reader_test_result"],
        "excluded_days_W": w["excluded_days_W"],
        "excluded_days_D": d["excluded_days_D"],
        "instrument_spec": d["instrument_spec"],
        "sigma_ref_bp": w["sigma_ref_bp"],
        "vol_quantiles_bp": w["vol_quantiles_bp"],
        "chosen_cell": d.get("chosen_cell") or "none",
        "outright_peak": d["outright_peak"],
        "admission": admission,
        "neighbourhood": d.get("neighbourhood", []),
        "sigma_d": d.get("sigma_d", {}),
        "mde": d.get("mde", {}),
        "compute": compute,
    }


def entry_text(day: date, values: dict[str, Any], frozen_sha: str) -> str:
    """The amendment's text: every frozen value, one line each, and the hash line."""
    lines = [f"### {day.isoformat()}: the D amendment", ""]
    lines.append("Values frozen on W and D, written into the configuration's placeholders and")
    lines.append("nothing else. Every number here is development: selected, biased upward.")
    lines.append("")
    for key, value in values.items():
        lines.append(f"- `{key}`: `{json.dumps(value, sort_keys=True, default=str)}`")
    lines += ["", f"Frozen configuration sha256: `{frozen_sha}`", ""]
    return "\n".join(lines)


def amend(repo: Path | str, values: dict[str, Any], day: date) -> str:
    root = Path(repo)
    config_path, document_path = root / config.CONFIG_PATH, root / config.DOCUMENT_PATH
    frozen = config.freeze_text(config_path.read_text(), values)
    config_path.write_text(frozen)
    sha = sha256_bytes(frozen.encode())
    document = document_path.read_text().rstrip("\n") + "\n\n" + entry_text(day, values, sha)
    document_path.write_text(document)
    return sha
