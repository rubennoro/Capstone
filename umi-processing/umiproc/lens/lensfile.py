"""The lens file (schema ``umiproc.lens.v1``): a measured ultrawide lens model, written by ``umiproc
lens-calibrate`` to ``<root>/lens/<pi_name>/lens.yaml`` and, with ``--apply``, copied to
``<root>/rigs/<rig_id>.lens.yaml`` (the optional lens file the processing reads instead of the rig's nominal model)::

    schema: umiproc.lens.v1
    rig_id: rig_pi_v3
    camera: ultrawide
    sensor: imx219
    resolution: [1280, 960]
    model:                          # the rig file's model block: umiproc.dp.geometry.camera_model.model_from_dict
      type: fisheye-equidistant     # Kannala-Brandt = cv2.fisheye = ORB-SLAM3 KannalaBrandt8 (umiproc.lens.fit)
      intrinsics: [fx, fy, cx, cy]  # pixels, pixel-centre convention (pixel (i, j) centred at u=j, v=i)
      distortion: [k1, k2, k3, k4]
    provenance: measured
    date: '2026-11-16T18:00:00Z'    # UTC, when it was fitted
    source: {episode_dir: ..., pi_name: ..., manifest_sha256: ..., camera_file: cam0.mp4}
    board: {...}                    # umiproc.lens.board.board_info()
    metrics: {...}                  # the values of the checks (report.json has everything)
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

__all__ = ["LENS_SCHEMA", "LensFileError", "LensFile", "lens_document", "dump_lens", "load_lens", "apply_lens"]

LENS_SCHEMA = "umiproc.lens.v1"
_HEADER = ("# Measured ultrawide lens model (umiproc lens-calibrate; DIRECTIONS.md section 2). Do not edit: re-run\n"
           "# the calibration instead. Schema and conventions: umiproc/lens/lensfile.py.\n")


class LensFileError(ValueError):
    """Invalid lens file. The message starts with the file name."""


@dataclass(frozen=True, eq=False)
class LensFile:
    rig_id: str
    camera: str
    sensor: str
    resolution: tuple[int, int]
    model: object                  # umiproc.dp.geometry.camera_model.FisheyeEquidistant
    provenance: str
    date: str
    source: dict
    metrics: dict
    path: str | None = None


def lens_document(*, rig_id: str, sensor: str, resolution, intrinsics, distortion, date: str, source: dict,
                  board: dict, metrics: dict) -> dict:
    """The lens file's content as a mapping (key order as written)."""
    return {"schema": LENS_SCHEMA, "rig_id": rig_id, "camera": "ultrawide", "sensor": sensor,
            "resolution": [int(resolution[0]), int(resolution[1])],
            "model": {"type": "fisheye-equidistant", "intrinsics": [float(v) for v in intrinsics],
                      "distortion": [float(v) for v in distortion]},
            "provenance": "measured", "date": date, "source": dict(source), "board": dict(board),
            "metrics": dict(metrics)}


class _Dumper(yaml.SafeDumper):
    """Block-style mappings, flow-style lists (``intrinsics: [fx, fy, cx, cy]``)."""


_Dumper.add_representer(list, lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=True))


def dump_lens(doc: dict) -> str:
    return _HEADER + yaml.dump(doc, Dumper=_Dumper, sort_keys=False, default_flow_style=False, width=1000)


def load_lens(path: str | os.PathLike) -> LensFile:
    """Load and validate a lens file (schema, resolution, a valid fisheye-equidistant model)."""
    from umiproc.dp.geometry.camera_model import FisheyeEquidistant, model_from_dict

    path = Path(path)
    try:
        d = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise LensFileError(f"{path}: cannot read ({e})") from None
    if not isinstance(d, dict):
        raise LensFileError(f"{path}: expected a mapping")
    if d.get("schema") != LENS_SCHEMA:
        raise LensFileError(f"{path}: schema: expected {LENS_SCHEMA!r}, got {d.get('schema')!r}")
    res = d.get("resolution")
    if (not isinstance(res, list) or len(res) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in res)):
        raise LensFileError(f"{path}: resolution: expected [width, height], got {res!r}")
    try:
        model = model_from_dict(d.get("model"), (res[0], res[1]))
    except ValueError as e:
        raise LensFileError(f"{path}: model.{e}") from None
    if not isinstance(model, FisheyeEquidistant):
        raise LensFileError(f"{path}: model.type: expected 'fisheye-equidistant', got {model.kind!r}")
    if d.get("provenance") != "measured":
        raise LensFileError(f"{path}: provenance: expected 'measured', got {d.get('provenance')!r}")
    for key in ("rig_id", "camera", "sensor", "date"):
        if not isinstance(d.get(key), str) or not d.get(key):
            raise LensFileError(f"{path}: {key}: expected a non-empty string, got {d.get(key)!r}")
    return LensFile(rig_id=d["rig_id"], camera=d["camera"], sensor=d["sensor"], resolution=(res[0], res[1]),
                    model=model, provenance=d["provenance"], date=d["date"], source=dict(d.get("source") or {}),
                    metrics=dict(d.get("metrics") or {}), path=str(path))


def _utc_stamp(now: _dt.datetime) -> str:
    return now.astimezone(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def apply_lens(lens_yaml: Path, dest: Path, now: _dt.datetime | None = None) -> tuple[Path, Path | None, bool]:
    """Copy ``lens_yaml`` to ``dest`` (the rig's lens file). An existing, different ``dest`` is first renamed to
    ``<dest>.bak-<UTC YYYYmmddTHHMMSSZ>``. Returns ``(dest, backup or None, changed)``; ``changed`` is False when
    ``dest`` already had exactly this content (nothing is written then)."""
    lens_yaml, dest = Path(lens_yaml), Path(dest)
    load_lens(lens_yaml)                                  # never apply a file that does not load
    data = lens_yaml.read_bytes()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.read_bytes() == data:
        return dest, None, False
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(data)
    backup = None
    if os.path.lexists(dest):
        base = dest.with_name(f"{dest.name}.bak-{_utc_stamp(now or _dt.datetime.now(_dt.timezone.utc))}")
        backup, k = base, 2
        while os.path.lexists(backup):
            backup = base.with_name(f"{base.name}-{k}")
            k += 1
        os.rename(dest, backup)
    os.replace(tmp, dest)
    return dest, backup, True
