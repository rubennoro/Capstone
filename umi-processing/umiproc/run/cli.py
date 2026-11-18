"""``umiproc run TASK_FOLDER START_DIR END_DIR [--source /data/incoming] [--root DIR] [--retry-failed]
[--dry-run]``: every demonstration in the range localized against the folder's map, checked, and finished into
``episode_N`` folders (:mod:`umiproc.run.command`; DIRECTIONS.md sections 5 and 7). Progress and the summary on stdout,
errors on stderr. Exit codes: 0 done (also with failed recordings: they are listed), 3 some wait for the gripper
calibration, 2 the command could not run (nothing changed), 1 an unexpected error stopped it."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from umiproc.run.command import EXIT_FAILED, EXIT_INPUT, EXIT_WAITING, RunInputError, RunResult, run_range

__all__ = ["add_arguments", "run", "format_summary", "next_block", "REASON_HINTS", "RETRY_REASONS"]

#: What to change in the recording, per reason code (UMI's rules, the pose checks, the copied validate).
REASON_HINTS = {
    "stopped_lost": "SLAM could not follow the demo against the map for 30 frames (1 s): start looking at the mapped "
                    "workspace, stay inside the area the map covered, no fast swings, never cover the lens",
    "too_many_lost": "more than 5 frames (0.17 s) were not tracked: move more smoothly, no fast swings or blank "
                     "views; the first frames count too (start still, looking at the mapped workspace)",
    "too_few_tracked": "fewer than 30 tracked frames (1 s): the demo must be tracked for at least a second",
    "late_relocalization": "the demo did not find itself in the map within 0.5 s: start still, looking at the "
                           "mapped workspace from where the mapping clip went",
    "no_piece": "no tracked stretch of 12 frames (0.4 s) between losses",
    "new_map": "SLAM lost the map and started a new one: stay inside the mapped area, move more slowly",
    "slam_error": "not a recording problem: read processed/localize/umiproc_slam.log",
    "gravity": "the demo's trajectory disagrees with the IMU's gravity (checked once on all its tracked frames): "
               "smooth motion, no bumps; if it repeats on smooth demos, check the lens and camera-IMU rotation (map "
               "check M4)",
    "scale_mismatch": "the demo's metric scale (checked once on all its tracked frames) is off by more than 10 %: "
                      "the map's scale (map check M4) or a SLAM jump",
    "speed": "the gripper moved faster than 3 m/s: move slower (or a SLAM jump)",
    "speed_extreme": "faster than 7 m/s: a SLAM jump, not the hand",
    "orientation": "the SLAM orientation disagrees with gravity by more than 3 deg (as map check M4): not the "
                   "recording if the demo was smooth (lens model, camera-IMU rotation)",
    "tag_coverage": "the finger tags were seen in fewer than 80 % of the frames: keep both visible (no hand or "
                    "object over them)",
    "kept_fraction": "widths or main-camera frames were missing for over 20 % of the piece: keep the finger tags "
                     "visible",
    "piece_too_short": "after trimming to its usable part the piece covers fewer than 12 frames",
    "no_grid_rows": "the piece lies outside the recording's start/stop span",
}


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("folder", metavar="TASK_FOLDER", help="the session's task folder (the name `umiproc map` printed)")
    p.add_argument("start", metavar="START_DIR", help="first demo recording's directory name in --source")
    p.add_argument("end", metavar="END_DIR", help="last demo recording's directory name (inclusive; the map and "
                                                  "calibration recordings in the range are skipped)")
    p.add_argument("--source", default="/data/incoming", metavar="DIR",
                   help="where the Pi delivers recordings (default /data/incoming; only read)")
    p.add_argument("--root", default=None, metavar="DIR", help="data root (default $UMIPROC_ROOT or ~/data/umi)")
    p.add_argument("--retry-failed", action="store_true", help="run the recordings in failed/ again (from scratch: "
                                                               "a new localization)")
    p.add_argument("--dry-run", action="store_true", help="check the folder and its map, print what would happen; "
                                                          "change nothing")


def _hint_lines(outcomes) -> list[str]:
    codes = []
    for o in outcomes:
        for r in o.reasons:                      # "too_many_lost" or "piece 2: speed, speed_extreme"
            codes += [c.strip() for c in r.split(": ")[-1].split(",")]
        for p in o.pieces:
            codes += list(p.get("reasons") or [])
    out = []
    for c in dict.fromkeys(codes):
        if c in REASON_HINTS:
            out.append(f"  - {c}: {REASON_HINTS[c]}")
    return out


def format_summary(root, res: RunResult) -> str:
    from umiproc.run.recording import EPISODE, FAILED, SKIPPED, WAITING
    from umiproc.status import folder_status, format_folder_status

    plan = res.plan
    lines = ["", f"run summary: tasks/{plan.folder}, {plan.start}..{plan.end}"]
    quiet = [o for o in res.outcomes if o.quiet]
    w = max([len(o.name) for o in res.outcomes if not o.quiet] or [1])
    lines += [f"  {o.name:<{w}}  ->  {o.describe()}" for o in res.outcomes if not o.quiet]
    if quiet:
        lines.append(f"  ({len(quiet)} already episodes of this folder: {', '.join(o.name for o in quiet)})")
    n = {s: len(res.by_state(s)) for s in (EPISODE, FAILED, WAITING, SKIPPED)}
    n_slots = sum(len(o.slots) for o in res.by_state(EPISODE))
    lines.append(f"  {n[EPISODE]} recording(s) -> {n_slots} episode(s), {n[FAILED]} failed, {n[WAITING]} waiting for "
                 f"the gripper calibration, {n[SKIPPED]} skipped")
    if res.error:
        lines.append(f"  stopped: {res.error}")
    hl = _hint_lines(res.by_state(FAILED) + res.by_state(EPISODE))
    if hl:
        lines += ["", "what the reasons mean and what to change (DIRECTIONS.md section 5.3):"] + hl
    lines.append("")
    try:
        lines.append(format_folder_status(folder_status(Path(root), plan.folder)))
    except Exception as e:  # noqa: BLE001 - the summary must never hide the run's own outcome
        lines.append(f"(status unavailable: {e})")
    return "\n".join(lines)


#: Failure reasons that are not about the recording: after fixing the cause, ``--retry-failed`` is worth it.
RETRY_REASONS = {"slam_error"}


def next_block(root, res: RunResult, args) -> str:
    """The ``Next:`` block after a run (DIRECTIONS.md sections 5-8)."""
    from umiproc.dp.stages.calibration import GRIPPER_RANGE_FILE
    from umiproc.next_step import block, carried, command
    from umiproc.run.recording import FAILED

    folder, flags = res.plan.folder, carried(args)
    has_gripper = (Path(root) / "tasks" / folder / GRIPPER_RANGE_FILE).is_file()
    same = command("umiproc", "run", folder, args.start, args.end, *flags,
                   *(["--retry-failed"] if args.retry_failed else []))
    if args.dry_run:
        return block("the same command without --dry-run", "7.2" if has_gripper else "5.2", same)
    if res.exit_code == EXIT_FAILED:
        return block("run it again: what was done keeps its outcome", "7.4", same)
    retry = []
    if any(c in RETRY_REASONS for o in res.by_state(FAILED) for c in o.reasons):
        retry = [command("umiproc", "run", folder, args.start, args.end, *flags, "--retry-failed")]
    if not has_gripper:
        if res.exit_code == EXIT_WAITING:
            return block("record the gripper calibration clip and run it; then the waiting demos are finished by any "
                         "run of this folder", "6",
                         command("umiproc", "calibrate", folder, "<GRIPPER_EPISODE>", *flags), *retry)
        return block("no demo is waiting: read the reasons, then record new test demos and run them"
                     + (" (or, once the SLAM problem is fixed, retry the failed ones)" if retry else ""), "5.4",
                     command("umiproc", "run", folder, "<FIRST_TEST_DEMO>", "<LAST_TEST_DEMO>", *flags), *retry)
    return block("check the folder, then record and run more demos or, after the session, copy it to the laptop"
                 + (" (retry the failed ones once the SLAM problem is fixed)" if retry else ""), "7.3",
                 command("umiproc", "status", folder, *carried(args, source=False)), *retry)


def run(args) -> int:
    from umiproc.paths import data_root
    from umiproc.status import _root_source

    try:
        root = data_root(args.root)
    except ValueError as e:
        print(f"umiproc run: {e}", file=sys.stderr)
        return EXIT_INPUT

    def log(*a, **k):
        print(*a, **k, flush=True)

    log(f"run: data root {root} (from {_root_source(args.root)})")
    try:
        res = run_range(root, args.folder, args.start, args.end, source=Path(args.source),
                        retry_failed=args.retry_failed, dry_run=args.dry_run, log=log)
    except RunInputError as e:
        print(f"umiproc run: {e}", file=sys.stderr)
        print("  nothing was changed", file=sys.stderr)
        return EXIT_INPUT
    except Exception:  # noqa: BLE001 - unexpected: traceback, exit 1
        traceback.print_exc()
        return EXIT_FAILED
    if args.dry_run:
        from umiproc.dp.run.plan import format_plan

        print(format_plan(res.plan).replace("dataproc", "umiproc"))
        print(f"map/{res.map.pi_name}: up to date (fingerprint {res.fingerprint_hash[:12]}); --dry-run: nothing was "
              "changed")
        print(next_block(root, res, args))
        return 0
    print(format_summary(root, res))
    if res.error:
        print(f"umiproc run: stopped at {res.error}", file=sys.stderr)
    print(next_block(root, res, args), flush=True)
    return res.exit_code
