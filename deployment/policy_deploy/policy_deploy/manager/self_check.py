"""python -m policy_deploy.manager.self_check [--station FILE] [--mock] [--json] (STRUCTURE.md 9.2).

The manager's rows of `policy_deploy check`: settings in range, the operator script (when set), the worst-gap rule
against link.yaml watchdog_limit_s, exec_h (info, PLAN 13 backup) and clock.estimator (info). It opens no socket.
With --json: one JSON object per line {"name", "status", "detail"} (status pass|warn|fail|info); exits 0 unless it
crashes. The terminal row is the launcher's (this subprocess has no terminal).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from policy_deploy.contracts import StationError


def rows(station: Path | None, mock: bool) -> list[tuple[str, str, str]]:
    from policy_deploy.manager.scripted_operator import ScriptedOperator
    from policy_deploy.manager.settings import Settings
    try:
        s = Settings.load(station, mock)
    except (StationError, ValueError) as e:
        return [("settings", "fail", f"{e}: check station.yaml manager: and config/manager.yaml")]
    out = [("settings", "pass", f"{s.station_file} (manager section valid; PC {s.commands_endpoint})")]
    if s.operator_script is not None:
        try:
            ScriptedOperator.load(s.operator_script, sys.stderr, lambda: 0, s.operator_script_wait_s)
            out.append(("operator script", "pass", str(s.operator_script)))
        except Exception as e:
            out.append(("operator script", "fail", f"{e}: check manager.operator_script"))
    gap = s.worst_gap_s
    rule = (f"heartbeat {s.heartbeat_period_s:g} + c3 timeout {s.c3_timeout_s:g} + health {s.timeouts.health_s:g} "
            f"+ jitter {s.link_jitter_allowance_s:g} = {gap:.2f} s")
    if gap < s.watchdog_limit_s:
        out.append(("worst gap", "pass", f"{rule} < watchdog_limit_s {s.watchdog_limit_s:g} s (link.yaml)"))
    else:
        out.append(("worst gap", "fail", f"{rule} is not below watchdog_limit_s {s.watchdog_limit_s:g} s: raise "
                                         f"watchdog_limit_s in link.yaml or lower the laptop timeouts"))
    out.append(("exec_h", "info", f"exec_h {s.exec_h} (PLAN 13 backup: raise it if the policy is slow)"))
    out.append(("clock.estimator", "info", f"{s.clock.estimator} (link.yaml)"))
    return out


def emit(items: list[tuple[str, str, str]], as_json: bool, out: TextIO) -> None:
    for name, status, detail in items:
        if as_json:
            out.write(json.dumps({"name": name, "status": status, "detail": detail}) + "\n")
        else:
            out.write(f"{status.upper():5} {name}: {detail}\n")
    out.flush()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m policy_deploy.manager.self_check", description=__doc__.split("\n")[0])
    ap.add_argument("--station", type=Path, default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    emit(rows(a.station, a.mock), a.json, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
