"""Only the study's ledger issues an access beyond development: no other module of the
package touches the study access's issuer token or passes an issuer."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "trading_research"
#: The token's home, and the one module that issues it.
HOME = {"data/study_access.py", "pipeline/short_horizon_ledger.py"}
#: The market-making rounds' own access classes, each with its own module-level token.
OWN_TOKENS = {"market_making/round2.py", "market_making/prereg.py"}


def _findings(path: Path, relative: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    out = []
    for node in ast.walk(tree):
        line = f"{relative}:{getattr(node, 'lineno', 0)}"
        if isinstance(node, ast.Attribute) and node.attr in ("_ISSUER", "_Issuer"):
            out.append(f"{line} reads .{node.attr}")
        if isinstance(node, ast.ImportFrom) and any(
            a.name in ("_ISSUER", "_Issuer") for a in node.names
        ):
            out.append(f"{line} imports the issuer")
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "_ISSUER" in node.value
        ):
            out.append(f"{line} names the issuer in a string")
        if relative in OWN_TOKENS:
            continue
        if isinstance(node, ast.Call) and any(k.arg == "issuer" for k in node.keywords):
            out.append(f"{line} passes issuer=")
        if isinstance(node, ast.Attribute) and node.attr == "issuer":
            out.append(f"{line} reads .issuer")
    return out


def test_no_module_outside_the_ledger_touches_the_issuer() -> None:
    found = []
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        if relative not in HOME:
            found += _findings(path, relative)
    assert found == []


def test_the_check_finds_each_way_of_reaching_the_token(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from trading_research.data import study_access as sa\n"
        "from trading_research.data.study_access import _ISSUER\n"
        "a = sa.StudyAccess(g, 'first read', issuer=sa._ISSUER)\n"
        "b = getattr(sa, '_ISSUER')\n"
        "c = replace(a, issuer=a.issuer)\n",
        encoding="utf-8",
    )
    found = " ".join(_findings(probe, "probe.py"))
    for what in (
        "imports the issuer",
        "reads ._ISSUER",
        "passes issuer=",
        "names the issuer",
        "reads .issuer",
    ):
        assert what in found
    ledger = (SRC / "pipeline" / "short_horizon_ledger.py").read_text(encoding="utf-8")
    assert "issuer=sa._ISSUER" in ledger  # the ledger is where it is issued
