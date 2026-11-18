"""``umiproc map TASK MAP_DIR [--source /data/incoming] [--root DIR] [--rig rig_pi_v3] [--first-map | --longest-map]``: build and check the
session map from a mapping recording; creates the task folder unless a blocking check (M1, M5's sightings) failed;
the other checks only warn (DIRECTIONS.md §4, :mod:`umiproc.map.build`). Progress on stdout, the result block on
stdout (pass) or stderr (fail).

Exit codes: 0 the map was attached (possibly with warnings); 1 a blocking check failed (nothing created; the
recording can be run again); 2 the command could not run on this input (the message says why; nothing created).
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
    g = p.add_mutually_exclusive_group()
    g.add_argument("--first-map", action="store_true",
                   help="when the whole clip gives more than one map, cut the clip where the first map piece stopped "
                        "tracking and map and check only that part (the whole clip's run is kept in mapping_full/)")
    g.add_argument("--longest-map", action="store_true",
                   help="the default, named: when the whole clip gives more than one map, the clip is cut to the map "
                        "piece that tracked the longest and only that part is mapped and checked (the whole clip's "
                        "run is kept in mapping_full/); no flag is needed for this")


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


def _first_map_line(fm: dict) -> str:
    longest = fm.get("mode") == "longest"
    flag = "longest map (automatic)" if longest else "--first-map"
    if not fm.get("applied"):
        return f"{flag}: not cut ({fm.get('reason')})"
    p = fm["piece"]
    which = "the longest map piece" if longest else "the clip up to the first map piece's end"
    return (f"{flag}: the whole clip gave {fm['maps_with_keyframes']} maps with keyframes; checked only {which}: "
            f"{fm['frames']} of {fm['frames_full']} frames, {fm.get('start_s', 0.0):.2f}-{fm['end_s']:.2f} s "
            f"(the piece tracked {p['first_tracked_s']:.2f}-{p['last_tracked_s']:.2f} s; whole-clip run in "
            f"{fm['full_run']['log'].split('/')[0]}/)")


def format_outcome(o: MapOutcome, task: str, map_dir: str) -> str:
    rep = o.report or {}
    lines = []
    warned = [c.id for c in o.checks if c.warned]
    if o.passed:
        lines.append(f"map: PASSED -> tasks/{o.folder} (label {task}), map in {o.where}"
                     + (f" (warnings: {', '.join(warned)}; the test demos will show whether they matter)"
                        if warned else ""))
    else:
        failed = [c.id for c in o.checks if c.failed]
        lines.append(f"map: FAILED for {map_dir}" + (f" ({o.pi_name})" if o.pi_name else "") +
                     (f": {', '.join(failed)} failed" if failed else "") + f" - {o.message}")
    for c in o.checks:
        if c.ok:
            state = "PASS"
        elif c.passed is None:
            state = "n/a "
        else:
            state = "WARN" if c.warned else "FAIL"
        lines.append(f"  {c.id} {state}  {c.title}: {c.message}")
        if not c.ok and c.hints:
            lines.append("       what to change:" if c.failed else "       to improve it (a warning, it does not block):")
            lines += [f"         - {h}" for h in c.hints]
    fm = rep.get("first_map") or rep.get("longest_map")
    if fm:
        lines.append("  " + _first_map_line(fm))
    if rep.get("diagnostics"):
        lines.append("  diagnostics (reported, not checked):")
        lines += [f"    {x}" for x in _diagnostic_lines(rep)]
        lines += [f"  WARNING: {w}" for w in rep["diagnostics"].get("warnings", [])]
    if o.where is not None:
        logs = [str(p) for p in (o.where / MAPPING_DIR / "umiproc_slam.log", o.where / RELOCALIZE_DIR /
                                 "umiproc_slam.log") if p.is_file()]
        lines.append(f"  report: {o.where / REPORT_NAME}" + (f"; SLAM logs: {', '.join(logs)}" if logs else ""))
        what = o.folder if o.passed and o.folder else str(o.where)
        if (o.where / MAPPING_DIR).is_dir():
            lines.append(f"  to look at the runs: umiproc slam-video {what} [--run all], umiproc slam-3d {what} "
                         "(into " + (f"tasks/{o.folder}/analysis/" if o.passed and o.folder else
                                     f"{o.where / 'analysis'}/") + ")")
    if o.passed:
        lines += ["", "Next: remove the table tag, record 2-3 test demos of the task, then run them against the map "
                  "(DIRECTIONS.md section 5):", f"  umiproc run {o.folder} <first_demo_dir> <last_demo_dir>"]
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
        o = build_map(root, args.task, args.map_dir, source=Path(args.source), rig_id=args.rig, log=log,
                      first_map=args.first_map, longest_map=args.longest_map)
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
