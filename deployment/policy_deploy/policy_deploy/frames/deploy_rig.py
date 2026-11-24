"""The deployment rig file: training's rig as the Pi streams it (policy_deploy/DIRECTIONS.md "Camera constants").

``python -m policy_deploy.frames.deploy_rig [--rig FILE] [--size W H] [--out FILE] [--check]``

Training (umi-processing) reads ``<data root>/rigs/<rig_id>.yaml`` and, when it exists, swaps the measured ultrawide
lens of ``<rig_id>.lens.yaml`` (same folder) into it (umiproc/sensors.py ``_apply_lens``). The Pi records at the
rig's sizes but streams both cameras at 640x480 (PI_PAYLOAD.md), the same sensor modes scaled, not cropped. This
writes one file in the same format (``dataproc.rig.v1``, parsed by the same loader as training's,
``_dataproc/config.py``) with:

* the lens folded in (``cameras.ultrawide.model`` and ``provenance: measured``), with training's checks: the lens
  file is for this rig, the ultrawide, its sensor and its recorded resolution;
* both cameras scaled to the stream size: ``f' = f * s``, ``c' = (c + 0.5) * s - 0.5``, distortion unchanged
  (scale-free), ``slam_mask`` points (pixel edges) times ``s``; refused when the aspect ratio differs (a crop is not a
  scale);
* ``rig_id`` ``<rig_id>_<W>``; everything else (virtual ultrawide, TCP, IMU, tags, gripper) unchanged.

Default output: ``deployment/contracts/rigs/<rig_id>_<W>.yaml``. ``--check`` writes nothing: exit 0 if that file is
what the sources give now, 1 if it differs (regenerate it), 2 if the sources are missing.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import yaml

DEFAULT_RIG = Path("~/data/umi/rigs/rig_pi_v3.yaml")
DEFAULT_SIZE = (640, 480)                                    # PI_PAYLOAD.md: both cameras 640x480 @ 30
CONTRACT_RIGS = Path(__file__).resolve().parents[3] / "contracts" / "rigs"
LENS_SCHEMA = "umiproc.lens.v1"


class DeployRigError(ValueError):
    """A source file that can't be used; names the file and the problem."""


def lens_path(rig_path: Path) -> Path:
    """Where training looks for the lens: ``<rig_id>.lens.yaml`` next to the rig file (umiproc/paths.py)."""
    return rig_path.with_name(rig_path.stem + ".lens.yaml")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scale_intrinsics(intr: Sequence[float], s: float) -> list[float]:
    fx, fy, cx, cy = (float(v) for v in intr)
    return [fx * s, fy * s, (cx + 0.5) * s - 0.5, (cy + 0.5) * s - 0.5]


def fold_lens(rig: dict, lens: dict, lens_file: Path) -> dict:
    """``rig`` with the lens file's model in ``cameras.ultrawide`` (training's _apply_lens checks first)."""
    if lens.get("schema") != LENS_SCHEMA:
        raise DeployRigError(f"{lens_file}: schema {lens.get('schema')!r}, expected {LENS_SCHEMA!r}")
    cam = rig["cameras"]["ultrawide"]
    problems = []
    if lens.get("rig_id") != rig["rig_id"]:
        problems.append(f"it is for rig {lens.get('rig_id')}")
    if lens.get("camera") != "ultrawide":
        problems.append(f"it is for camera {lens.get('camera')!r}")
    if str(lens.get("sensor", "")).lower() != str(cam["sensor"]).lower():
        problems.append(f"it is for sensor {lens.get('sensor')}, the rig's ultrawide is {cam['sensor']}")
    if list(lens.get("resolution", [])) != list(cam["resolution"]):
        problems.append(f"it was measured at {lens.get('resolution')}, the rig's ultrawide records {cam['resolution']}")
    if problems:
        raise DeployRigError(f"{lens_file}: " + "; ".join(problems))
    out = copy.deepcopy(rig)
    m = lens["model"]
    out["cameras"]["ultrawide"]["model"] = {"type": m["type"], "intrinsics": [float(v) for v in m["intrinsics"]],
                                            "distortion": [float(v) for v in m["distortion"]]}
    out["cameras"]["ultrawide"]["provenance"] = "calibrated"   # the rig schema's word (training's report says calibrated)
    return out


def scale_to(rig: dict, size: tuple[int, int]) -> dict:
    """Both cameras as delivered at ``size`` (the same view, scaled)."""
    out = copy.deepcopy(rig)
    w2, h2 = size
    for role, cam in out["cameras"].items():
        w, h = (int(v) for v in cam["resolution"])
        if w * h2 != h * w2:
            raise DeployRigError(f"cameras.{role}: {w}x{h} can't be scaled to {w2}x{h2} (another aspect ratio "
                                 "is a crop, not a scale)")
        s = w2 / w
        cam["resolution"] = [w2, h2]
        cam["model"]["intrinsics"] = _scale_intrinsics(cam["model"]["intrinsics"], s)
        if "slam_mask" in cam:
            cam["slam_mask"]["polygons"] = [[[float(x) * s, float(y) * s] for x, y in poly]
                                            for poly in cam["slam_mask"]["polygons"]]
    out["rig_id"] = f"{rig['rig_id']}_{w2}"
    return out


class _Dumper(yaml.SafeDumper):
    pass


def _list(dumper: yaml.SafeDumper, data: list) -> Any:
    flow = all(not isinstance(v, (list, dict)) for v in data)      # numbers on one line; a matrix row per line
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _list)


def _home(path: Path) -> str:
    """``path`` with the home folder written as ~ (the header is the same on every machine)."""
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def render(rig_path: Path, size: tuple[int, int]) -> tuple[str, str]:
    """(rig_id, file text) for the deployment rig made from ``rig_path`` (+ its lens file, if any)."""
    rig_path = rig_path.expanduser()
    if not rig_path.is_file():
        raise FileNotFoundError(f"{rig_path}: no rig file (training's data root: ~/data/umi/rigs/)")
    rig = yaml.safe_load(rig_path.read_text())
    lens_file = lens_path(rig_path)
    lines = [
        "# GENERATED by `python -m policy_deploy.frames.deploy_rig`: do not edit. Regenerate after the sources change",
        "# (a new lens calibration, an edited rig file). policy_deploy/DIRECTIONS.md \"Camera constants\".",
        f"# rig:  {_home(rig_path)}  sha256 {_sha256(rig_path)}",
    ]
    if lens_file.is_file():
        lens = yaml.safe_load(lens_file.read_text())
        rig = fold_lens(rig, lens, lens_file)
        lines.append(f"# lens: {_home(lens_file)}  sha256 {_sha256(lens_file)}  (measured {lens.get('date')})")
    else:
        lines.append(f"# lens: none ({_home(lens_file)} absent): the ultrawide keeps the rig's own model, as training does")
    out = scale_to(rig, size)
    lines.append(f"# Both cameras scaled from the recorded sizes to the {size[0]}x{size[1]} stream (f' = f*s, "
                 "c' = (c+0.5)*s-0.5); nothing else changed.")
    body = yaml.dump(out, Dumper=_Dumper, sort_keys=False, width=120, allow_unicode=True)
    return out["rig_id"], "\n".join(lines) + "\n" + body


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m policy_deploy.frames.deploy_rig", description=__doc__.split("\n")[0])
    p.add_argument("--rig", type=Path, default=DEFAULT_RIG, help=f"training's rig file (default {DEFAULT_RIG})")
    p.add_argument("--size", type=int, nargs=2, default=DEFAULT_SIZE, metavar=("W", "H"),
                   help="the stream size of both cameras (default 640 480)")
    p.add_argument("--out", type=Path, default=None, help="default: deployment/contracts/rigs/<rig_id>_<W>.yaml")
    p.add_argument("--check", action="store_true", help="compare with the file instead of writing it")
    a = p.parse_args(argv)
    try:
        rig_id, text = render(a.rig, tuple(a.size))
    except (OSError, DeployRigError, KeyError, TypeError, yaml.YAMLError) as e:
        print(f"deploy_rig: FAIL {e}")
        return 2
    out = a.out or CONTRACT_RIGS / f"{rig_id}.yaml"
    if a.check:
        if out.is_file() and out.read_text() == text:
            print(f"deploy_rig: OK {out} matches its sources")
            return 0
        print(f"deploy_rig: DIFFERS {out} is not what the sources give now; regenerate: "
              "python -m policy_deploy.frames.deploy_rig")
        return 1
    from policy_deploy.frames._dataproc.config import ConfigError
    from policy_deploy.frames.rig_file import load_live_rig
    with tempfile.TemporaryDirectory() as tmp:          # it must load exactly as the frames process loads it
        probe = Path(tmp) / out.name
        probe.write_text(text)
        try:
            load_live_rig(probe)
        except ConfigError as e:
            print(f"deploy_rig: FAIL the generated file does not load: {e}")
            return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"deploy_rig: wrote {out} ({rig_id}, both cameras {a.size[0]}x{a.size[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
