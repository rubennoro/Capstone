"""``dataproc calibrate``: progress on stdout, the result (or the failure block on stderr), exit code 0 / 1."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from dataproc.calibrate.diagnostics import format_failure, format_stats
from dataproc.calibrate.task import CalibrateResult, calibrate_task
from dataproc.paths import DataRoot
from dataproc.pi_intake import PiIntakeError

__all__ = ["EXIT_OK", "EXIT_FAILED", "cli_calibrate", "format_success"]

EXIT_OK, EXIT_FAILED = 0, 1


def _rig(root: DataRoot, rig_id: str):
    """The rig yaml for the hints, or None when it cannot be loaded."""
    try:
        from dataproc.config import load_rig
        return load_rig(root.rig_file(rig_id))
    except Exception:  # noqa: BLE001 - the hints then name the fields without their values
        return None


def format_success(res: CalibrateResult, source: Path) -> str:
    """Folder, measured opening, tag statistics, warnings and the next command."""
    fit = res.fit
    o = fit.details.get("opening", {})
    expected = (f" (expected {o['expected_m']:.3f} m ± {o['tolerance_m']:.3f} m)"
                if o.get("expected_m") is not None else " (not checked: the rig yaml has no gripper section)")
    lines = [f"calibrate: OK -> tasks/{res.folder} (label {res.task}, rig {res.rig_id}), calibration {res.slot}",
             f"  measured opening {fit.opening_m:.3f} m{expected}; min_width {o['min_width']:.4f} m, max_width "
             f"{o['max_width']:.4f} m",
             *format_stats(fit.diagnostics)]
    lines += [f"  warning: {w}" for w in fit.warnings]
    src = "" if Path(source) == Path("/data/incoming") else f" --source {source}"
    lines += ["", "Next: process the demonstrations of this session (directory names in the source, inclusive):",
              f"  dataproc run {res.folder} <start_dir> <end_dir>{src}"]
    return "\n".join(lines)


def cli_calibrate(root: DataRoot, task: str, episode_dir: str, source="/data/incoming",
                  rig_id: str = "rig_pi_v2") -> int:
    """``dataproc calibrate TASK EPISODE_DIR [--source DIR] [--rig RIG]``: returns the exit code."""
    try:
        res = calibrate_task(root, task, episode_dir, source=Path(source), rig_id=rig_id, log=print)
    except (PiIntakeError, ValueError) as e:
        print(f"dataproc calibrate: {e}", file=sys.stderr)
        return EXIT_FAILED
    except BlockingIOError:
        print(f"dataproc calibrate: another dataproc calibrate/run holds {root.path}/.run.lock; wait for it",
              file=sys.stderr)
        return EXIT_FAILED
    except Exception:  # noqa: BLE001 - unexpected: traceback, exit 1
        traceback.print_exc()
        return EXIT_FAILED
    if not res.ok:
        print(format_failure(res, _rig(root, rig_id)), file=sys.stderr)
        return EXIT_FAILED
    print(format_success(res, Path(source)))
    return EXIT_OK
