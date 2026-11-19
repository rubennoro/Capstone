# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/launcher/check_result.py @ 2ffe641
# changes: the command and process names in comments and the docstring; the section number. The source's own
# header follows.
# copied from rnoro/deployment:deployment/deploy/deploy/check_result.py @ 91b178b
# changes: none (docstring names)
"""CheckResult, one line of the `teleop_collect check` table, and the table printer (STRUCTURE.md 10.1)."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TextIO

STATUSES = ("pass", "warn", "fail", "info")


@dataclass(frozen=True)
class CheckResult:
    process: str          # "fake_pc", "frames", "teleop", "ui", or "station" for the shared files
    name: str             # short, e.g. "checkpoint"
    status: str           # "pass" | "warn" | "fail" | "info"
    detail: str           # one line: what was found, and on failure what to check

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"CheckResult status must be one of {STATUSES}, got {self.status!r}")


def print_table(results: Sequence[CheckResult], out: TextIO) -> None:
    """One row per result, in the given order, with aligned columns: process, check, STATUS, detail."""
    header = ("process", "check", "status", "detail")
    rows = [(r.process, r.name, r.status.upper(), " ".join(r.detail.split())) for r in results]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(3)]
    for row in [header, tuple("-" * w for w in widths) + ("-" * len(header[3]),), *rows]:
        out.write("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row[:3])) + "  " + row[3] + "\n")


def exit_code(results: Sequence[CheckResult]) -> int:
    """2 if any check failed, else 0 (warn and info never stop a run)."""
    return 2 if any(r.status == "fail" for r in results) else 0
