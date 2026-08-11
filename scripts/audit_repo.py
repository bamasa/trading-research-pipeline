"""Disclosure audit for the public repository.

This project was written from scratch, but it was informed by closed-source work.
The checks below are the automated half of that boundary: they fail the build if
anything that belongs on the private side of the line shows up in the working
tree. They run in CI on every push, not just before the first release, because a
leak is much cheaper to catch in review than after it is public.

What is checked:

1. No absolute filesystem paths that identify a machine, a user or an internal
   mount point.
2. No credential-shaped strings.
3. No references to private libraries or internal tooling.
4. No notebook outputs (they embed data, paths and figures).
5. No heavy or binary artifacts (model weights, datasets, pickles).
6. The ignore rules for generated directories are actually in place.

Run with ``make audit`` or ``uv run python scripts/audit_repo.py``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# This file necessarily contains every pattern it searches for, so it is the one
# file that cannot audit itself.
SELF = Path(__file__).resolve()

# Extensions skipped by the text scan, because they hold no readable text.
#
# An allowlist of "text extensions" was tried first and was the wrong shape for
# a security check: anything not on the list went unscanned in silence, and the
# first thing that fell through was a ``.log`` file carrying absolute local
# paths. A check that fails open is worse than no check, because it is trusted.
# So the default is now to read every file, and only known-binary formats are
# skipped — a new extension is scanned automatically rather than ignored.
SKIP_TEXT_SCAN = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".svg",
        ".ico",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".so",
        ".dylib",
        ".dll",
        ".bin",
        ".lock",  # uv.lock: machine-generated, thousands of hashes, no prose
    }
)

# Extensions that must never be committed regardless of size.
FORBIDDEN_SUFFIXES = frozenset(
    {
        ".parquet",
        ".feather",
        ".arrow",
        ".h5",
        ".hdf5",
        ".pkl",
        ".pickle",
        ".joblib",
        ".onnx",
        ".pth",
        ".pt",
        ".ckpt",
        ".npy",
        ".npz",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
    }
)

MAX_FILE_BYTES = 1_000_000

# Directories that must be ignored by git so that generated output cannot be
# committed by accident.
MUST_BE_IGNORED = ("data", "artifacts", "models", "runs", "reports")


@dataclass(frozen=True)
class Rule:
    """A single forbidden pattern.

    ``allow`` lists substrings that make a match legitimate. Keeping the
    exceptions next to the rule means an exception has to be argued for once, in
    the open, rather than by quietly loosening the pattern.
    """

    name: str
    pattern: re.Pattern[str]
    reason: str
    allow: tuple[str, ...] = ()


RULES: tuple[Rule, ...] = (
    Rule(
        name="internal-mount-path",
        pattern=re.compile(r"/mnt/\S+"),
        reason="absolute path to an internal mount point",
    ),
    Rule(
        name="home-directory-path",
        pattern=re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/\S*"),
        reason="absolute path that identifies a machine or account",
    ),
    Rule(
        name="credential-assignment",
        # Only flag an assignment to a non-empty literal. A bare mention of the
        # word "token" in prose is not a leak; `api_key = "abc123"` is.
        pattern=re.compile(
            r"""(?ix)
            \b(api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token
              |password|passwd|credential[s]?|private[_-]?key)\b
            \s*[:=]\s*
            ["'][^"'\s]{6,}["']
            """
        ),
        reason="looks like a hard-coded credential",
    ),
    Rule(
        name="private-key-block",
        pattern=re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        reason="embedded private key",
    ),
    Rule(
        name="aws-access-key",
        pattern=re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        reason="AWS access key id",
    ),
    Rule(
        name="private-dependency",
        pattern=re.compile(r"(?i)\b(crypto_runner|prog_trader|stock_info_parser)\b"),
        reason="reference to a private internal library or binary",
    ),
    Rule(
        name="internal-account",
        pattern=re.compile(r"(?i)\bquant0\d\b"),
        reason="internal account identifier",
    ),
    Rule(
        name="email-address",
        pattern=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        reason="personal contact details",
        # The generic security contact in the policy file is deliberate.
        allow=("security@example.com",),
    ),
    Rule(
        name="private-host",
        pattern=re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        reason="hard-coded IP address",
        # Version strings and the loopback address in local dev instructions.
        allow=("127.0.0.1", "0.0.0.0"),
    ),
)


class Finding(list):
    """Collected audit failures, grouped for readable output."""


# Directories the fallback walk must never descend into. Third-party packages
# and generated data would otherwise bury real findings under thousands of
# irrelevant ones — vendored licence files are full of contributor email
# addresses, and every wheel ships test fixtures in forbidden formats.
FALLBACK_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "htmlcov",
        "data",
        "artifacts",
        "models",
        "runs",
        "reports",
    }
)


def tracked_files() -> list[Path]:
    """Return files git would include, falling back to a filesystem walk.

    Using git's own view is the point: it is exactly the set that would be
    published, and it respects ``.gitignore`` without reimplementing it here.

    The fallback exists so the audit still runs before ``git init`` or in an
    exported tree, but it is strictly worse: it approximates the ignore rules
    with a fixed list, so a real repository should always take the git path.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("note: not a git repository; falling back to a filesystem walk")
        return [
            p
            for p in REPO_ROOT.rglob("*")
            if p.is_file() and not FALLBACK_SKIP_DIRS.intersection(p.relative_to(REPO_ROOT).parts)
        ]
    return [REPO_ROOT / line for line in out.stdout.splitlines() if line]


def scan_text(path: Path, findings: Finding) -> None:
    """Apply every pattern rule to one text file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return

    rel = path.relative_to(REPO_ROOT)
    for rule in RULES:
        for match in rule.pattern.finditer(text):
            hit = match.group(0)
            if any(allowed in hit for allowed in rule.allow):
                continue
            line_no = text.count("\n", 0, match.start()) + 1
            # The matched text is echoed truncated: enough to locate the
            # problem, not enough to republish a secret in the build log.
            excerpt = hit[:40] + ("…" if len(hit) > 40 else "")
            findings.append(f"{rel}:{line_no}: [{rule.name}] {rule.reason}: {excerpt!r}")


def scan_notebook(path: Path, findings: Finding) -> None:
    """Fail on notebooks that carry saved outputs or execution counts."""
    rel = path.relative_to(REPO_ROOT)
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        findings.append(f"{rel}: notebook could not be parsed ({exc})")
        return

    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("outputs"):
            findings.append(f"{rel}: cell {i} has saved outputs; clear them before committing")
        if cell.get("execution_count") is not None:
            findings.append(f"{rel}: cell {i} has an execution count; clear it before committing")


def scan_binary(path: Path, findings: Finding) -> None:
    """Fail on forbidden extensions and oversized files."""
    rel = path.relative_to(REPO_ROOT)
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        findings.append(f"{rel}: forbidden file type '{path.suffix}'; data and weights stay local")
        return
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        findings.append(
            f"{rel}: {size / 1_000_000:.1f} MB exceeds the {MAX_FILE_BYTES / 1_000_000:.0f} MB limit"
        )


def check_ignore_rules(findings: Finding) -> None:
    """Confirm the generated-output directories are ignored, and anchored.

    The anchoring matters more than it looks. An unanchored ``data/`` pattern
    matches at every level, so it also excludes ``src/lobml/data/`` — the
    package's own source. That failure is silent: tests still pass locally
    because the files are on disk, and the repository ships without them.
    """
    gitignore = REPO_ROOT / ".gitignore"
    if not gitignore.exists():
        findings.append(".gitignore: missing")
        return

    text = gitignore.read_text(encoding="utf-8")
    for name in MUST_BE_IGNORED:
        if re.search(rf"^/{re.escape(name)}/\s*$", text, re.MULTILINE):
            continue
        if re.search(rf"^{re.escape(name)}/\s*$", text, re.MULTILINE):
            findings.append(
                f".gitignore: '{name}/' is unanchored and matches at every level; "
                f"write '/{name}/' so it cannot exclude package source"
            )
        else:
            findings.append(f".gitignore: '/{name}/' is not ignored")


def check_sources_are_tracked(paths: list[Path], findings: Finding) -> None:
    """Confirm the package source actually made it into the repository.

    An audit that only looks for things that should be absent will happily pass
    an empty repository. This is the other half: the files the project cannot
    work without must be present in git's own view of what would be published.
    """
    src = REPO_ROOT / "src"
    if not src.is_dir():
        return

    on_disk = {p for p in src.rglob("*.py") if "__pycache__" not in p.parts}
    visible = set(paths)
    missing = sorted(p.relative_to(REPO_ROOT) for p in on_disk - visible)
    for path in missing:
        findings.append(
            f"{path}: package source exists on disk but is excluded from the repository"
        )


def iter_scannable(paths: list[Path]) -> Iterator[Path]:
    for path in paths:
        if not path.is_file() or path.resolve() == SELF:
            continue
        yield path


def main() -> int:
    findings = Finding()
    paths = list(iter_scannable(tracked_files()))

    for path in paths:
        scan_binary(path, findings)
        if path.suffix.lower() not in SKIP_TEXT_SCAN:
            scan_text(path, findings)
        if path.suffix == ".ipynb":
            scan_notebook(path, findings)

    check_ignore_rules(findings)
    check_sources_are_tracked(paths, findings)

    print(f"disclosure audit: scanned {len(paths)} files")
    if findings:
        print(f"\n{len(findings)} problem(s) found:\n")
        for item in findings:
            print(f"  {item}")
        print("\nFix these before committing. See docs/disclosure_policy.md.")
        return 1
    print("no problems found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
