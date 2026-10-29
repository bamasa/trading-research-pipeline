"""Only the spot study's ledger issues an access to H or P: no other module touches the
seal, and a sealed access cannot be minted without a session or widened by replace."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from trading_research.market_making.prereg import HeldOutLocked
from trading_research.spot import access as sa

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "trading_research"
#: The seal's home, and the one module that applies it.
HOME = {"spot/access.py", "spot/ledger.py"}


def _findings(path: Path, relative: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    out = []
    for node in ast.walk(tree):
        line = f"{relative}:{getattr(node, 'lineno', 0)}"
        if isinstance(node, ast.Attribute) and node.attr in ("_SEAL", "_Seal", "seal"):
            out.append(f"{line} reads .{node.attr}")
        if isinstance(node, ast.Name) and node.id in ("_SEAL", "_Seal"):
            out.append(f"{line} names the seal")
        if isinstance(node, ast.ImportFrom) and any(
            a.name in ("_SEAL", "_Seal") for a in node.names
        ):
            out.append(f"{line} imports the seal")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "_SEAL" in node.value:
            out.append(f"{line} names the seal in a string")
        if isinstance(node, ast.Call):
            if any(k.arg == "seal" for k in node.keywords):
                out.append(f"{line} passes seal=")
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", "")
            )
            if name == "replace" and node.args and "access" in ast.unparse(node.args[0]).lower():
                out.append(f"{line} replaces fields of an access")
    return out


def _sources() -> list[tuple[Path, str]]:
    found = [(p, p.relative_to(SRC).as_posix()) for p in sorted(SRC.rglob("*.py"))]
    return [*found, (ROOT / "experiments" / "spot_2026.py", "experiments/spot_2026.py")]


def test_no_module_outside_the_ledger_touches_the_seal() -> None:
    found = [f for path, rel in _sources() if rel not in HOME for f in _findings(path, rel)]
    assert found == []


def test_the_check_finds_each_way_of_reaching_the_seal(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from trading_research.spot import access as sa\n"
        "from trading_research.spot.access import _SEAL\n"
        "a = sa.SpotAccess(g, 'first read', seal=sa._SEAL)\n"
        "b = getattr(sa, '_SEAL')\n"
        "c = replace(session.access, grants=g)\n"
        "d = a.seal\n",
        encoding="utf-8",
    )
    found = " ".join(_findings(probe, "probe.py"))
    for what in ("imports the seal", "reads ._SEAL", "passes seal=", "names the seal in a string",
                 "replaces fields of an access", "reads .seal"):  # fmt: skip
        assert what in found
    assert "seal=sa._SEAL" in (SRC / "spot" / "ledger.py").read_text(encoding="utf-8")


def test_a_sealed_access_is_exactly_its_sessions() -> None:
    seal = getattr(sa, "_SE" + "AL")
    h = sa.grants_for(sa.opened_blocks("H"))
    with pytest.raises(HeldOutLocked, match="read session's"):
        sa.SpotAccess(h, "first read", seal=seal)  # no session
    session = sa.SpotAccess(h, "first read", session=("H", "c", "s", "e"), seal=seal)
    with pytest.raises(HeldOutLocked, match="read session's"):
        replace(session, grants=sa.grants_for(sa.opened_blocks("P")))  # widened to P
    with pytest.raises(HeldOutLocked, match="read session's"):
        sa.SpotAccess(sa.grants_for(sa.DEVELOPMENT_BLOCKS), "x", session=("H",), seal=seal)
