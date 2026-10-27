"""The synthetic carry study's entry point, for the harness's demonstration.

``python -m agents.workflows.research_loop demo`` copies this file into a
scratch repository as ``experiments/synth_carry.py``, as the owner's code
between registration and run. Every daily series it computes on is drawn from
the seeded generator its registration names: nothing is fetched and no
market's data is read, so nothing it prints is evidence about a market. It
keeps the registered studies' shape in miniature: the D stage fills the null
placeholder and appends the amendment; ``open H`` needs the amendment
committed and appends the ledger's open entry; ``read H`` needs that entry
committed, writes the results file, records its sha256 in the ledger and
writes the page with the registered headline sentence filled in::

    python -m experiments.synth_carry status
    python -m experiments.synth_carry D
    python -m experiments.synth_carry open H
    python -m experiments.synth_carry read H
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG = "configs/synth_carry.yaml"
DOCUMENT = "docs/preregistration/synth_carry.md"
LEDGER = "experiments/results/synth_carry_ledger.json"
RESULTS = "experiments/results/synth_carry_H.json"
PAGE = "experiments/results/synth_carry_page.md"
PLACEHOLDER = re.compile(r"^(\s*sigma_d_bp: )null(  # frozen by the D amendment)$", re.MULTILINE)
KILLED = (
    "C1 on H, synthetic venue: {status}; first kill {kill} at {mean:.2f} bp against {ref:.2f} bp."
)
UNKILLED = "C1 on H, synthetic venue: {status}; no kill fired; mean {mean:.2f} bp a day, t {t:.2f}."


class Refused(RuntimeError):
    """A stage out of order, or a block the ledger has not opened."""


def git(*args: str) -> str | None:
    done = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    return done.stdout if done.returncode == 0 else None


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def config(text: str | None = None) -> dict[str, Any]:
    loaded = yaml.safe_load(text if text is not None else (REPO / CONFIG).read_text())
    return loaded if isinstance(loaded, dict) else {}


def ledger(committed: bool) -> dict[str, list[dict[str, Any]]]:
    text = git("show", f"HEAD:{LEDGER}") if committed else None
    if not committed and (REPO / LEDGER).exists():
        text = (REPO / LEDGER).read_text()
    return json.loads(text) if text else {}


def series(cfg: dict[str, Any], block: str) -> list[float]:
    """The block's daily net in bp of notional, from the registered seed."""
    span, gen = cfg["blocks"][block], cfg["generator"]
    draw = random.Random(gen[f"seed_{block}"])
    days = (span["end"] - span["start"]).days + 1
    return [draw.gauss(gen["drift_bp"], gen["sd_bp"]) for _ in range(days)]


def status() -> dict[str, Any]:
    frozen = config()["frozen"]["sigma_d_bp"] is not None
    committed = config(git("show", f"HEAD:{CONFIG}") or "").get("frozen", {})
    disk, head = ledger(False).get("H", []), ledger(True).get("H", [])
    h = "closed"
    if disk:
        h = "opened, not committed" if not head else "open"
        h = "read" if disk[-1].get("results_sha256") else h
    return {
        "study": "synth_carry",
        "D": "amended" if frozen else "not run",
        "amendment_committed": committed.get("sigma_d_bp") is not None,
        "H": h,
    }


def stage_d() -> dict[str, Any]:
    text = (REPO / CONFIG).read_text()
    if not PLACEHOLDER.search(text):
        raise Refused("D has run: its amendment is in the configuration")
    sigma = round(statistics.stdev(series(config(), "D")), 4)
    frozen = PLACEHOLDER.sub(rf"\g<1>{sigma}\g<2>", text)
    (REPO / CONFIG).write_text(frozen)
    line = f"- {datetime.now(UTC).date()}, the D amendment: sigma_d_bp = {sigma}. "
    line += f"Frozen configuration sha256: `{sha256(frozen)}`\n"
    with (REPO / DOCUMENT).open("a") as document:
        document.write("\n" + line)
    return {
        "stage": "D",
        "sigma_d_bp": sigma,
        "frozen_sha256": sha256(frozen),
        "next": "the owner commits the D amendment before H is opened",
    }


def open_h() -> dict[str, Any]:
    if not status()["amendment_committed"]:
        raise Refused("H opens only once the D amendment is committed")
    if ledger(False).get("H"):
        raise Refused("H has an open entry already")
    if (git("status", "--porcelain") or "").strip():
        raise Refused("H opens only on a clean tree")
    entry = {
        "block": "H",
        "commit": (git("rev-parse", "HEAD") or "").strip(),
        "config_sha256": sha256((REPO / CONFIG).read_text()),
        "read": "first read",
        "results_sha256": None,
        "utc": datetime.now(UTC).isoformat(),
    }
    (REPO / LEDGER).parent.mkdir(parents=True, exist_ok=True)
    (REPO / LEDGER).write_text(json.dumps({"H": [entry]}, indent=2, sort_keys=True) + "\n")
    return {"opened": "H", "ledger": LEDGER, "next": "the owner commits the open entry"}


def read_h() -> dict[str, Any]:
    entries = ledger(True).get("H", [])
    if not entries:
        raise Refused("H is locked: its open entry is not committed")
    if entries[-1].get("results_sha256"):
        raise Refused("H has been read: a read is never run twice")
    if sha256((REPO / CONFIG).read_text()) != entries[-1]["config_sha256"]:
        raise Refused("the configuration is not the one H was opened with")
    cfg = config()
    values = series(cfg, "H")
    mean = statistics.fmean(values)
    t = mean / (statistics.stdev(values) / math.sqrt(len(values)))
    positive = sum(v > 0 for v in values) / len(values)
    kill = "K1" if mean <= cfg["kills"]["K1_reference_bp"] else None
    if kill is None and positive <= cfg["kills"]["K_days_share"]:
        kill = "K-days"
    state = (
        "killed"
        if kill
        else ("candidate" if t >= cfg["statuses"]["candidate_t"] else "inconclusive")
    )
    slot = {
        "status": state,
        "first_kill": kill,
        "mean_bp": round(mean, 2),
        "t": round(t, 2),
        "positive_days": round(positive, 4),
        "days": len(values),
    }
    text = json.dumps({"block": "H", "synthetic": True, "statuses": {"C1": slot}}, indent=2) + "\n"
    (REPO / RESULTS).write_text(text)
    disk = ledger(False)
    disk["H"][-1]["results_sha256"] = sha256(text)
    (REPO / LEDGER).write_text(json.dumps(disk, indent=2, sort_keys=True) + "\n")
    ref = cfg["kills"]["K1_reference_bp"]
    sentence = (KILLED if kill else UNKILLED).format(
        status=state, kill=kill, mean=mean, ref=ref, t=t
    )
    page = [
        "# synth_carry: pages",
        "",
        "Synthetic: drawn from a seeded generator; not evidence about any market.",
        "",
        "## H",
        "",
        "### Headline sentences",
        "",
        f"- {sentence}",
        "",
    ]
    (REPO / PAGE).write_text("\n".join(page))
    return {
        "read": "H",
        "results": RESULTS,
        "results_sha256": sha256(text),
        "ledger": LEDGER,
        "page": PAGE,
        "statuses": {"C1": slot},
    }


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser(
        prog="python -m experiments.synth_carry",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    found.add_argument("command", choices=["status", "D", "open", "read"])
    found.add_argument("block", nargs="?", choices=["H"])
    return found


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    actions = {"status": status, "D": stage_d, "open": open_h, "read": read_h}
    if args.command in {"open", "read"} and args.block != "H":
        parser().error("open and read name the block: H")
    try:
        record = actions[args.command]()
    except Refused as exc:
        print(f"refused: {exc}")
        return 1
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
