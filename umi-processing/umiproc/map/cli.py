"""``umiproc map TASK MAP_DIR [--source /data/incoming] [--root DIR] [--rig rig_pi_v3]``: build and check the
session map from a mapping recording; creates the task folder only when every check passed (DIRECTIONS.md §3,
:mod:`umiproc.map.build`). Progress on stdout, the result block on stdout (pass) or stderr (fail).

Exit codes: 0 every check passed; 1 a check failed (nothing created; the recording can be run again); 2 the command
could not run on this input (the message says why; nothing created).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from umiproc.map.build import (EXIT_FAILED, EXIT_INPUT, MAPPING_DIR, RELOCALIZE_DIR, REPORT_NAME, MapInputError,
                               MapOutcome, build_map)

__all__ = ["add_arguments", "run", "format_outcome"]


def add_arguments(p: argparse.ArgumentParser) -> None:
    from umiproc.sensors import DEFAULT_RIG

    p.add_argument("task", metavar="TASK", help="the session's task label, e.g. block_raise: creates tasks/TASK "
                                                "(TASK_2, ... when taken) when the map passes")
    p.add_argument("map_dir", metavar="MAP_DIR", help="the mapping recording's directory name in --source, "
                                                      "e.g. episode_000123")
    p.add_argument("--source", default="/data/incoming", metavar="DIR",
                   help="where the Pi delivers recordings (default /data/incoming; only read)")
    p.add_argument("--root", default=None, metavar="DIR", help="data root (default $UMIPROC_ROOT or ~/data/umi)")
    p.add_argument("--rig", default=DEFAULT_RIG, help=f"rig id (default {DEFAULT_RIG})")


def _fmt_pacing(name: str, p: dict | None) -> str:
    if not p:
        return f"{name}: no pacing figures"
    return (f"{name}: fed {p['feed_rate_hz']:.1f} of {p['recording_rate_hz']:.1f} frames/s, tracking "
            f"{p['track_ms_median']:.0f}/{p['track_ms_p95']:.0f}/{p['track_ms_max']:.0f} ms (median/p95/max), "
            f"{p['frames_over_period']} frames over their period")


def _diagnostic_lines(rep: dict) -> list[str]:
    d = rep.get("diagnostics") or {}
    out = []
    b = d.get("build") or {}
    v = d.get("vio_inflation") or {}
    lens = d.get("lens") or {}
    out.append(f"SLAM build: {b.get('binary')} (patches {b.get('patches_label')})")
    out.append(f"IMU vio_inflation: noise_density {v.get('noise_density')}, random_walk {v.get('random_walk')}")
    lens_s = lens.get("provenance")
    if lens.get("date"):
        lens_s += f" {lens['date']}"
    out.append(f"lens: {lens_s}")
    out.append(f"fingerprint: {str(rep.get('fingerprint_hash'))[:12]}")
    out.append(f"keyframes in the map: {d.get('keyframes')}")
    out.append(f"tag frames: tag seen in {d.get('tag_frames')} of {d.get('frames')} frames")
    t = d.get("time_offset_check")
    if t and t.get("traj_delay_ms") is not None:
        out.append(f"camera-IMU time offset estimate: {t['traj_delay_ms']:.1f} ms (correlation {t['corr']:.2f}; "
                   "umiproc uses 0, this is only reported)")
    elif t:
        out.append(f"camera-IMU time offset estimate: none ({t.get('reason')})")
    for name, p in (d.get("pacing") or {}).items():
        out.append("pacing, " + _fmt_pacing(name, p))
    return out


def format_outcome(o: MapOutcome, task: str, map_dir: str) -> str:
    rep = o.report or {}
    lines = []
    if o.passed:
        lines.append(f"map: PASSED -> tasks/{o.folder} (label {task}), map in {o.where}")
    else:
        failed = [c.id for c in o.checks if not c.ok]
        lines.append(f"map: FAILED for {map_dir}" + (f" ({o.pi_name})" if o.pi_name else "") +
                     (f": {', '.join(failed)} failed" if failed else "") + f" - {o.message}")
    for c in o.checks:
        state = "PASS" if c.ok else ("n/a " if c.passed is None else "FAIL")
        lines.append(f"  {c.id} {state}  {c.title}: {c.message}")
        if not c.ok and c.hints:
            lines.append("       what to change:")
            lines += [f"         - {h}" for h in c.hints]
    if rep.get("diagnostics"):
        lines.append("  diagnostics (reported, not checked):")
        lines += [f"    {x}" for x in _diagnostic_lines(rep)]
        lines += [f"  WARNING: {w}" for w in rep["diagnostics"].get("warnings", [])]
    if o.where is not None:
        logs = [str(p) for p in (o.where / MAPPING_DIR / "umiproc_slam.log", o.where / RELOCALIZE_DIR /
                                 "umiproc_slam.log") if p.is_file()]
        lines.append(f"  report: {o.where / REPORT_NAME}" + (f"; SLAM logs: {', '.join(logs)}" if logs else ""))
    if o.passed:
        lines += ["", "Next: remove the table tag, record 2-3 test demos of the task, then run them against the map "
                  "(DIRECTIONS.md section 3.3-3.5):", f"  umiproc run {o.folder} <first_demo_dir> <last_demo_dir>"]
    elif o.exit_code == EXIT_FAILED and o.checks:
        lines.append(f"  nothing was attached and {map_dir} was not taken: change what the hints say and run "
                     f"`umiproc map {task} {map_dir}` again (or record a new mapping clip)")
    return "\n".join(lines)


def run(args) -> int:
    from umiproc.paths import data_root

    try:
        root = data_root(args.root)
    except ValueError as e:
        print(f"umiproc map: {e}", file=sys.stderr)
        return EXIT_INPUT
    from umiproc.status import _root_source

    def log(*a, **k):
        print(*a, **k, flush=True)

    log(f"map: data root {root} (from {_root_source(args.root)})")
    try:
        o = build_map(root, args.task, args.map_dir, source=Path(args.source), rig_id=args.rig, log=log)
    except MapInputError as e:
        print(f"umiproc map: {e}", file=sys.stderr)
        print("  nothing was created and the recording was not taken", file=sys.stderr)
        return EXIT_INPUT
    except Exception:  # noqa: BLE001 - unexpected: traceback, exit 1
        traceback.print_exc()
        return EXIT_FAILED
    text = format_outcome(o, args.task, args.map_dir)
    sys.stdout.flush()
    print(text, file=sys.stdout if o.passed else sys.stderr, flush=True)
    return o.exit_code
