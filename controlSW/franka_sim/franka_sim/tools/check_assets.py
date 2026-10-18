"""Verify the vendored assets against assets/SOURCE.md (existence, sha256, no unlisted extra files).

Why: the menagerie models are pinned by commit; any silent edit or stray file must be caught.
Usage: ``python -m franka_sim check-assets [--assets DIR]`` (exit 0 ok, 1 on any mismatch).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

DEFAULT_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_ROW = re.compile(r"^\|\s*`?([^|`]+?)`?\s*\|\s*(\d+)\s*\|\s*([0-9a-f]{64})\s*\|\s*$")


def parse_source(source_md: Path) -> dict[str, tuple[int, str]]:
    """Return {relative path: (bytes, sha256)} from the SOURCE.md table."""
    out: dict[str, tuple[int, str]] = {}
    for line in source_md.read_text().splitlines():
        m = _ROW.match(line)
        if m:
            out[m.group(1)] = (int(m.group(2)), m.group(3))
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(assets: Path) -> tuple[int, list[str]]:
    """Return (number of listed files, list of problems)."""
    source = assets / "SOURCE.md"
    if not source.is_file():
        return 0, [f"missing {source}"]
    listed = parse_source(source)
    problems: list[str] = []
    if not listed:
        problems.append("SOURCE.md lists no files")
    for rel, (size, digest) in sorted(listed.items()):
        p = assets / rel
        if not p.is_file():
            problems.append(f"missing: {rel}")
        elif p.stat().st_size != size:
            problems.append(f"size mismatch: {rel} ({p.stat().st_size} != {size})")
        elif sha256_file(p) != digest:
            problems.append(f"sha256 mismatch: {rel}")
    for p in sorted(assets.rglob("*")):
        if p.is_file():
            rel = p.relative_to(assets).as_posix()
            if rel != "SOURCE.md" and rel not in listed:
                problems.append(f"extra file not in SOURCE.md: {rel}")
    return len(listed), problems


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--assets", default=str(DEFAULT_ASSETS), help="assets dir (default: the packaged one)")


def run(args: argparse.Namespace) -> int:
    n, problems = check(Path(args.assets))
    if problems:
        print(f"check-assets FAILED: {len(problems)} problem(s) in {n} listed files")
        for s in problems:
            print("  " + s)
        return 1
    print(f"check-assets OK: {n} files match SOURCE.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
