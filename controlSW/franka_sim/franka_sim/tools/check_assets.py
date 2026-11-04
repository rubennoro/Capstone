"""Verify the vendored assets against assets/SOURCE.md (existence, sha256, no unlisted extra files).

Why: the menagerie models are pinned by commit; any silent edit or stray file must be caught.
Finger kits under ``assets/finger_kits/<id>/`` (A9) each need their own SOURCE.md (written by ``finger-kit pin``): every
listed file must match, kit.yaml's listed files must all be in it, no unlisted extras, and its ``kit sha256`` line must
equal ``finger_kit.kit_digest``. They are excluded from the top-level "extra file" scan.
Usage: ``python -m franka_sim check-assets [--assets DIR]`` (exit 0 ok, 1 on any mismatch).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

KITS_DIR = "finger_kits"
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
            if rel.startswith(KITS_DIR + "/"):
                continue                                     # checked per kit by check_kits
            if rel != "SOURCE.md" and rel not in listed:
                problems.append(f"extra file not in SOURCE.md: {rel}")
    problems += check_kits(assets)[1]
    return len(listed), problems


def check_kits(assets: Path) -> tuple[int, list[str]]:
    """Verify every ``assets/finger_kits/*/SOURCE.md`` against its kit directory. Returns (kits checked, problems)."""
    from franka_sim import finger_kit as FK

    root = assets / KITS_DIR
    problems: list[str] = []
    kits = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    for kd in kits:
        tag = f"{KITS_DIR}/{kd.name}"
        src = kd / "SOURCE.md"
        if not src.is_file():
            problems.append(f"{tag}: missing SOURCE.md (run: python -m franka_sim finger-kit pin {tag})")
            continue
        listed = parse_source(src)
        for rel, (size, digest) in sorted(listed.items()):
            p = kd / rel
            if not p.is_file():
                problems.append(f"{tag}: missing: {rel}")
            elif p.stat().st_size != size:
                problems.append(f"{tag}: size mismatch: {rel} ({p.stat().st_size} != {size})")
            elif sha256_file(p) != digest:
                problems.append(f"{tag}: sha256 mismatch: {rel}")
        for p in sorted(kd.rglob("*")):
            rel = p.relative_to(kd).as_posix()
            if p.is_file() and rel != "SOURCE.md" and rel not in listed:
                problems.append(f"{tag}: extra file not in SOURCE.md: {rel}")
        try:
            want = {FK.KIT_YAML, *FK.listed_files(kd)}
            for rel in sorted(want - set(listed)):
                problems.append(f"{tag}: kit file not in SOURCE.md: {rel}")
            digest = FK.kit_digest(kd)
            line = [ln for ln in src.read_text().splitlines() if ln.startswith("- kit sha256:")]
            if not line or line[0].split(":", 1)[1].strip() != digest:
                problems.append(f"{tag}: SOURCE.md kit sha256 differs from the kit digest {digest}")
        except (FK.KitError, OSError) as e:
            problems.append(f"{tag}: {e}")
    return len(kits), problems


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--assets", default=str(DEFAULT_ASSETS), help="assets dir (default: the packaged one)")


def run(args: argparse.Namespace) -> int:
    n, problems = check(Path(args.assets))
    nk = check_kits(Path(args.assets))[0]
    if problems:
        print(f"check-assets FAILED: {len(problems)} problem(s) in {n} listed files")
        for s in problems:
            print("  " + s)
        return 1
    print(f"check-assets OK: {n} files match SOURCE.md" + (f"; {nk} finger kit(s) verified" if nk else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
