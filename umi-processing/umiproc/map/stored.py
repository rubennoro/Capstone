"""Reading a folder's map back (``umiproc status``; step 3's ``run``): its report, whether the current settings and
build still match it, and the failed maps kept in ``<root>/maps_failed/``. Read only.

**Out of date**: the fingerprint the map was made with (``map/<pi_name>/slam_fingerprint.json``) differs from the one
the current data-root files (rig file, SLAM config, lens file), the current mask/settings and the current
umiproc_slam build (``$UMIPROC_SLAM_BINARY`` or the default, with its ``BUILD_INFO`` patches and library) give. The
map and every episode localized against it then have to be made again (docs/DESIGN.md "SLAM").
"""

from __future__ import annotations

import tempfile
from pathlib import Path

__all__ = ["MapInfo", "read_map", "current_fingerprint", "fingerprint_differences", "failed_maps"]


class MapInfo(dict):
    """``pi_name``, ``dir``, ``complete`` (report.json there), ``passed``, ``created_utc``, ``fingerprint_hash``,
    ``checks`` ({id: passed}), ``build`` and ``vio_inflation`` (from the report), ``record`` (the stored
    fingerprint record or None), ``work_report`` (for an incomplete map: ``<root>/.work/map/<pi_name>/report.json``
    when an interrupted pass left it there, else None)."""


def read_map(task_dir: Path) -> list[MapInfo]:
    """The map slots of a task folder (one, normally), each with what its report and fingerprint say."""
    from umiproc.dp.files import read_json
    from umiproc.map.build import FINGERPRINT_NAME, REPORT_NAME, WORK_DIR
    from umiproc.recordings import map_slots

    out = []
    root = Path(task_dir).parent.parent                  # <root>/tasks/<folder>
    for slot in map_slots(task_dir):
        rep = read_json(slot / REPORT_NAME)
        fp = read_json(slot / FINGERPRINT_NAME)
        work_report = root / WORK_DIR / "map" / slot.name / REPORT_NAME
        info = MapInfo(pi_name=slot.name, dir=slot, complete=isinstance(rep, dict), passed=None, created_utc=None,
                       fingerprint_hash=None, checks={}, build=None, vio_inflation=None,
                       record=fp.get("record") if isinstance(fp, dict) else None,
                       work_report=work_report if not isinstance(rep, dict) and work_report.is_file() else None)
        if isinstance(rep, dict):
            d = rep.get("diagnostics") or {}
            info.update(passed=rep.get("passed"), created_utc=rep.get("created_utc"),
                        fingerprint_hash=rep.get("fingerprint_hash"),
                        checks={c.get("id"): c.get("passed") for c in rep.get("checks", [])},
                        build=d.get("build"), vio_inflation=d.get("vio_inflation"))
        out.append(info)
    return out


def current_fingerprint(root: Path, rig_id: str, binary=None) -> tuple[str, dict]:
    """``(hash, record)`` of the fingerprint a ``map`` run would store now. The settings file and mask are written to
    a temporary folder outside the root (the hash depends on their bytes, not where they are). Raises the errors of
    :func:`umiproc.sensors.load_sensors` (``ensure=False``: nothing is copied) and
    :class:`umiproc.slam_inputs.SlamBuildError`."""
    from umiproc.sensors import load_sensors
    from umiproc.slam_inputs import write_slam_inputs

    sensors = load_sensors(Path(root), rig_id, log=lambda *a, **k: None, ensure=False)
    with tempfile.TemporaryDirectory(prefix="umiproc-fp-") as tmp:
        inputs = write_slam_inputs(sensors, Path(tmp), binary)
        return inputs.fingerprint_hash, inputs.fingerprint


def fingerprint_differences(stored: dict | None, current: dict) -> list[str]:
    """Dotted keys (two levels deep) where two fingerprint records differ, e.g. ``sensors.imu``, ``binary``."""
    if not isinstance(stored, dict):
        return ["(no stored fingerprint)"]
    out = []
    for k in sorted(set(stored) | set(current)):
        a, b = stored.get(k), current.get(k)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            out += [f"{k}.{s}" for s in sorted(set(a) | set(b)) if a.get(s) != b.get(s)]
        else:
            out.append(k)
    return out


def failed_maps(root: Path) -> list[dict]:
    """Every ``maps_failed/<pi_name>/<UTC>/``, newest first: ``{pi_name, when, dir, failed (check ids), error}``."""
    from umiproc.dp.files import read_json
    from umiproc.map.build import FAILED_DIR, REPORT_NAME

    base = Path(root) / FAILED_DIR
    out = []
    if not base.is_dir():
        return out
    for rec in sorted(p for p in base.iterdir() if p.is_dir()):
        for run in sorted((p for p in rec.iterdir() if p.is_dir()), reverse=True):
            rep = read_json(run / REPORT_NAME, {}) or {}
            failed = [c.get("id") for c in rep.get("checks", []) if c.get("passed") is not True]
            out.append({"pi_name": rec.name, "when": run.name, "dir": run, "failed": failed,
                        "error": "unexpected error" if rep.get("error") else None})
    out.sort(key=lambda d: d["when"], reverse=True)
    return out
