"""The SLAM settings file ``configs/slam/<rig_id>.yaml`` (schema ``umiproc.slam.v1``), docs/DESIGN.md "SLAM".

::

    schema: umiproc.slam.v1
    rig: rig_pi_v3
    mask:                       # the gripper mask (UMI's mechanism: set to 0 in every frame fed to ORB-SLAM3)
      source: "traced ..."      # optional free-form note
      resolution: [1280, 960]   # the ultrawide's native size the polygons are drawn in
      polygons:                 # >= 1 polygon, each >= 3 [x, y] points enclosing >= 1 square pixel
        - [[x, y], [x, y], [x, y], ...]
    table_tag: {dictionary: DICT_4X4_50, id: 13, size_m: 0.16}
    orb: {n_features: 1500, scale_factor: 1.2, n_levels: 8, ini_th_fast: 20, min_th_fast: 7}

Mask coordinates are pixel **edges** (``0 <= x <= W``, ``0 <= y <= H``), the convention of
:mod:`umiproc.dp.slam.mask`, which draws them (:func:`mask_keep`). Unknown keys are errors; every error is a
:class:`SlamConfigError` whose message starts with ``<file>: <yaml.path>:``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

__all__ = ["SLAM_SCHEMA", "SlamConfigError", "MaskConfig", "TableTag", "OrbSettings", "SlamConfig",
           "load_slam_config", "parse_slam_config", "mask_keep"]

SLAM_SCHEMA = "umiproc.slam.v1"


class SlamConfigError(ValueError):
    """Invalid SLAM settings file. The message starts with ``<file>: <yaml.path>:``."""


@dataclass(frozen=True)
class MaskConfig:
    """``mask``: polygons as float ``(x, y)`` pixel edges at ``resolution = (W, H)``."""

    resolution: tuple[int, int]
    polygons: tuple[tuple[tuple[float, float], ...], ...]
    source: str | None = None


@dataclass(frozen=True)
class TableTag:
    """``table_tag``: the ArUco tag on the table (OpenCV predefined dictionary name, marker id, black-square side)."""

    dictionary: str
    id: int
    size_m: float


@dataclass(frozen=True)
class OrbSettings:
    """``orb``: ORB-SLAM3's ``ORBextractor.*`` settings."""

    n_features: int
    scale_factor: float
    n_levels: int
    ini_th_fast: int
    min_th_fast: int


@dataclass(frozen=True)
class SlamConfig:
    schema: str
    rig: str
    mask: MaskConfig
    table_tag: TableTag
    orb: OrbSettings
    source_path: str | None = None


# --------------------------------------------------------------------------------------------- reader

class _Map:
    """A mapping read key by key; :meth:`finish` rejects keys that were never read."""

    def __init__(self, d: Any, path: str, src: str):
        self.src, self.path = src, path
        if not isinstance(d, dict):
            raise SlamConfigError(f"{src}: {path or '<top>'}: expected a mapping, got {type(d).__name__}")
        self.d, self.used = d, set()

    def p(self, key: str) -> str:
        return f"{self.path}.{key}" if self.path else key

    def err(self, key: str, msg: str) -> SlamConfigError:
        return SlamConfigError(f"{self.src}: {self.p(key)}: {msg}")

    def req(self, key: str) -> Any:
        self.used.add(key)
        if key not in self.d:
            raise self.err(key, "missing")
        return self.d[key]

    def opt(self, key: str, default=None) -> Any:
        self.used.add(key)
        return self.d.get(key, default)

    def sub(self, key: str) -> "_Map":
        return _Map(self.req(key), self.p(key), self.src)

    def finish(self) -> None:
        unknown = sorted(str(k) for k in set(self.d) - self.used)
        if unknown:
            raise self.err(unknown[0], "unknown key")

    def int(self, key: str, positive: bool = False, min_value: int | None = None) -> int:
        v = self.req(key)
        if isinstance(v, bool) or not isinstance(v, int):
            raise self.err(key, f"expected an integer, got {v!r}")
        if positive and v <= 0:
            raise self.err(key, f"must be > 0, got {v}")
        if min_value is not None and v < min_value:
            raise self.err(key, f"must be >= {min_value}, got {v}")
        return v

    def float(self, key: str, positive: bool = False) -> float:
        return _number(self.req(key), self, key, positive)

    def str(self, key: str) -> str:
        v = self.req(key)
        if not isinstance(v, str) or not v:
            raise self.err(key, f"expected a non-empty string, got {v!r}")
        return v


def _number(v: Any, m: _Map, key: str, positive: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise m.err(key, f"expected a number, got {v!r}")
    f = float(v)
    if not math.isfinite(f):
        raise m.err(key, f"must be finite, got {v!r}")
    if positive and f <= 0:
        raise m.err(key, f"must be > 0, got {v!r}")
    return f


# --------------------------------------------------------------------------------------------- sections

def _mask(m: _Map) -> MaskConfig:
    res = m.req("resolution")
    if (not isinstance(res, (list, tuple)) or len(res) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in res)):
        raise m.err("resolution", f"expected [width, height] (positive integers), got {res!r}")
    w, h = int(res[0]), int(res[1])
    raw = m.req("polygons")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise m.err("polygons", f"expected a non-empty list of polygons, got {raw!r}")
    polygons = []
    for i, poly in enumerate(raw):
        key = f"polygons[{i}]"
        if not isinstance(poly, (list, tuple)) or len(poly) < 3:
            raise m.err(key, f"expected a polygon: a list of at least 3 points [x, y], got {poly!r}")
        pts = []
        for j, pt in enumerate(poly):
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                raise m.err(f"{key}[{j}]", f"expected a point [x, y] of 2 numbers, got {pt!r}")
            x, y = (_number(v, m, f"{key}[{j}][{k}]") for k, v in enumerate(pt))
            if not (0.0 <= x <= w and 0.0 <= y <= h):
                raise m.err(f"{key}[{j}]", f"point [{x:g}, {y:g}] is outside the {w}x{h} image (pixel edges: "
                            f"0 <= x <= {w}, 0 <= y <= {h})")
            pts.append((x, y))
        area = 0.5 * abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1])))
        if area < 1.0:
            raise m.err(key, f"degenerate polygon (area {area:g} square pixels, at least 1 needed)")
        polygons.append(tuple(pts))
    source = m.opt("source")
    if source is not None and not isinstance(source, str):
        raise m.err("source", f"expected a string, got {source!r}")
    m.finish()
    return MaskConfig((w, h), tuple(polygons), source or None)


def _table_tag(m: _Map) -> TableTag:
    import cv2

    name = m.str("dictionary")
    if not name.startswith("DICT_") or not isinstance(getattr(cv2.aruco, name, None), int):
        raise m.err("dictionary", f"not an OpenCV predefined ArUco dictionary: {name!r}")
    n_markers = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)).bytesList.shape[0]
    tag_id = m.int("id", min_value=0)
    if tag_id >= n_markers:
        raise m.err("id", f"{name} has ids 0..{n_markers - 1}, got {tag_id}")
    tag = TableTag(name, tag_id, m.float("size_m", positive=True))
    m.finish()
    return tag


def _orb(m: _Map) -> OrbSettings:
    orb = OrbSettings(n_features=m.int("n_features", positive=True), scale_factor=m.float("scale_factor", True),
                      n_levels=m.int("n_levels", positive=True), ini_th_fast=m.int("ini_th_fast", positive=True),
                      min_th_fast=m.int("min_th_fast", positive=True))
    if orb.scale_factor <= 1.0:
        raise m.err("scale_factor", f"must be > 1 (the ORB pyramid's scale step), got {orb.scale_factor:g}")
    if orb.min_th_fast > orb.ini_th_fast:
        raise m.err("min_th_fast", f"must be <= ini_th_fast ({orb.ini_th_fast}): ORB-SLAM3 lowers the FAST "
                    f"threshold from ini_th_fast to min_th_fast, got {orb.min_th_fast}")
    m.finish()
    return orb


def parse_slam_config(d: Any, src: str = "<slam config>") -> SlamConfig:
    """Validate a decoded SLAM settings document (``src`` names it in error messages)."""
    from umiproc.dp.paths import NAME_RE

    m = _Map(d, "", src)
    schema = m.str("schema")
    if schema != SLAM_SCHEMA:
        raise m.err("schema", f"expected {SLAM_SCHEMA!r}, got {schema!r}")
    rig = m.str("rig")
    if not NAME_RE.fullmatch(rig):
        raise m.err("rig", f"invalid rig id {rig!r}: use only letters, digits, '_' and '-'")
    cfg = SlamConfig(schema=schema, rig=rig, mask=_mask(m.sub("mask")), table_tag=_table_tag(m.sub("table_tag")),
                     orb=_orb(m.sub("orb")), source_path=src)
    m.finish()
    return cfg


def load_slam_config(path: str | os.PathLike) -> SlamConfig:
    """Load and validate a SLAM settings file. The file name up to its first dot must equal its ``rig``
    (``rig_pi_v3.yaml``, or e.g. ``rig_pi_v3.slam.yaml`` for a copy next to the rig file)."""
    path = Path(path)
    try:
        d = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise SlamConfigError(f"{path}: cannot read ({e.strerror or e})") from None
    except yaml.YAMLError as e:
        raise SlamConfigError(f"{path}: invalid YAML ({e})") from None
    cfg = parse_slam_config(d, str(path))
    stem = path.name.split(".", 1)[0]
    if stem != cfg.rig:
        raise SlamConfigError(f"{path}: rig: {cfg.rig!r} does not match the file name {stem!r}")
    return cfg


def mask_keep(slam_cfg: SlamConfig, size: tuple[int, int] | None = None) -> np.ndarray:
    """The gripper mask drawn at ``size = (w, h)`` (default: the mask's own resolution): uint8 ``(h, w)``,
    **255 = keep**, **0 = masked**, by :func:`umiproc.dp.slam.mask.keep_mask` (pixel-edge convention and the
    fill rule are documented there). Read-only (cached): copy it before modifying."""
    from umiproc.dp.slam.mask import keep_mask

    res = slam_cfg.mask.resolution
    return keep_mask(slam_cfg.mask.polygons, res, res if size is None else size)
