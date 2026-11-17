"""``umiproc calibrate TASK_FOLDER EPISODE_DIR [--source /data/incoming] [--root DIR]``: the gripper opening range of
a session folder (:mod:`umiproc.calibrate.gripper`; DIRECTIONS.md section 5). Progress on stdout, the result block
on stdout (pass) or stderr (fail). Exit codes: 0 passed; 1 failed (nothing recorded, the recording can be run
again); 2 the command could not run on this input."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

from umiproc.calibrate.gripper import (EXIT_FAILED, EXIT_INPUT, CalibrateInputError, CalibrateOutcome,
                                       calibrate_folder)

__all__ = ["add_arguments", "run", "format_outcome", "hints"]


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("folder", metavar="TASK_FOLDER", help="the session's task folder (the name `umiproc map` printed, "
                                                         "e.g. block_raise or block_raise_2)")
    p.add_argument("cal_dir", metavar="EPISODE_DIR", help="the gripper calibration recording's directory name in "
                                                          "--source, e.g. episode_000124")
    p.add_argument("--source", default="/data/incoming", metavar="DIR",
                   help="where the Pi delivers recordings (default /data/incoming; only read)")
    p.add_argument("--root", default=None, metavar="DIR", help="data root (default $UMIPROC_ROOT or ~/data/umi)")


def hints(o: CalibrateOutcome) -> list[str]:
    """What to change in the recording (or the setup), most specific first."""
    from umiproc.dp.calibrate.diagnostics import hints as dp_hints

    out = []
    if "finger_tags_below_80" in o.reasons:
        r = o.rule
        worse = "left" if r.get("left_rate", 0) <= r.get("right_rate", 0) else "right"
        out.append(f"keep both finger tags in view the whole time (the {worse} one was missed most): no hand, "
                   "glove or object over them, good even light, open and close slowly (motion blur), the lens in "
                   "focus. UMI's rule: each tag in at least 80 % of the frames")
    if o.fit is not None and any(c != "finger_tags_below_80" for c in o.reasons):
        res = SimpleNamespace(reasons=[c for c in o.reasons if c != "finger_tags_below_80"], fit=o.fit,
                              rig_id="")         # what the copied hints read
        out += [h.replace("dataproc", "umiproc") for h in dp_hints(res)]
    if "intake" in o.reasons:
        out.append("intake refused the recording (see the quarantine reason.json): record a new calibration clip")
    return list(dict.fromkeys(out))


def format_outcome(o: CalibrateOutcome, source: Path) -> str:
    from umiproc.dp.calibrate.diagnostics import format_stats

    fit = o.fit
    lines = []
    if o.passed:
        n = len(fit.warnings)
        lines.append(f"calibrate: PASSED -> tasks/{o.folder}/gripper_range.json (from {o.slot})"
                     + (f" - {n} warning{'s' if n > 1 else ''}, read them before `umiproc run`" if n else ""))
    else:
        lines.append(f"calibrate: FAILED for {o.source_name}" + (f" ({o.pi_name})" if o.pi_name else "")
                     + f" - {o.message}")
    if fit is not None and fit.details.get("opening"):
        op = fit.details["opening"]
        exp = (f" (expected {op['expected_m']:.3f} m +- {op['tolerance_m']:.3f} m)" if op.get("expected_m") is not None
               else "")
        closed = f" (expected {op['expected_closed_m']:.3f} m)" if op.get("expected_closed_m") is not None else ""
        lines.append(f"  measured opening {op['measured_m']:.3f} m{exp}; closed width {op['min_width']:.4f} m"
                     f"{closed}, open width {op['max_width']:.4f} m")
    if fit is not None:
        lines += [ln.replace("fit needs 20% of frames with each tag", "the fit needs 20 %, UMI's rule 80 %")
                  for ln in format_stats(fit.diagnostics)]
    if o.rule:
        lines.append(f"  UMI's finger-tag rule {'PASS' if o.rule['ok'] else 'FAIL'}: {o.rule['message']}")
    lines.append(f"  lens: {o.lens.get('provenance')}" + (f" {o.lens['date']}" if o.lens.get("date") else ""))
    if fit is not None:
        lines += [f"  warning: {w}" for w in fit.warnings]
    if o.replaced and o.passed:
        lines.append(f"  it replaces the earlier calibration {', '.join(o.replaced)} (no episode was made with it)")
    if not o.passed:
        h = hints(o)
        if h:
            lines.append("  what to change:")
            lines += [f"    - {x}" for x in h]
        if o.exit_code == EXIT_FAILED:
            lines.append(f"  nothing was recorded and {o.source_name} was not taken: record a new calibration clip "
                         f"(or fix the cause) and run `umiproc calibrate {o.folder} <episode_dir>` again")
    else:
        src = "" if Path(source) == Path("/data/incoming") else f" --source {source}"
        lines += ["", "Next: the demos (DIRECTIONS.md sections 4-5):",
                  f"  umiproc run {o.folder} <first_demo_dir> <last_demo_dir>{src}",
                  "  (demos already localized and waiting in .staging/ are finished by any run of this folder)"]
    return "\n".join(lines)


def run(args) -> int:
    from umiproc.paths import data_root
    from umiproc.status import _root_source

    try:
        root = data_root(args.root)
    except ValueError as e:
        print(f"umiproc calibrate: {e}", file=sys.stderr)
        return EXIT_INPUT

    def log(*a, **k):
        print(*a, **k, flush=True)

    log(f"calibrate: data root {root} (from {_root_source(args.root)})")
    try:
        o = calibrate_folder(root, args.folder, args.cal_dir, source=Path(args.source), log=log)
    except CalibrateInputError as e:
        print(f"umiproc calibrate: {e}", file=sys.stderr)
        print("  nothing was created and the recording was not taken", file=sys.stderr)
        return EXIT_INPUT
    except Exception:  # noqa: BLE001 - unexpected: traceback, exit 1
        traceback.print_exc()
        return EXIT_FAILED
    sys.stdout.flush()
    print(format_outcome(o, Path(args.source)), file=sys.stdout if o.passed else sys.stderr, flush=True)
    return o.exit_code
